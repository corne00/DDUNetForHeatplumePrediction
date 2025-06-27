import torch
import torch.nn as nn
import copy
from typing import Dict
from .sub_modules import CNNCommunicator, Encoder, Decoder

class MultiGPU_UNet_with_comm(nn.Module):
    # Initialize the network architecture
    # unet = model(communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
    #                                communication_network_def=communication_network, num_convs=num_convs)
    def __init__(self, settings: Dict, input_shape: tuple = (640,640), devices: list = ["cuda:0"], bilinear: bool = False, 
                 communicator_type = None, comm_network_but_no_communication = None, communication_network_def = CNNCommunicator):
        super(MultiGPU_UNet_with_comm, self).__init__()
        # init general part
        self.input_shape = input_shape
        self.devices = devices

        # init model
        self.kernel_size = settings["model"]["kernel_size"]
        self.padding = self.padding = settings["model"]["padding"] if settings["model"]["padding"] is not None else self.kernel_size // 2
        self.dropout_rate = settings["model"]["dropout_rate"]

        # init unet-specific part
        self.n_channels = settings["model"]["UNet"]["num_channels"]
        self.n_classes = settings["model"]["UNet"]["num_outputs"]
        self.depth = settings["model"]["UNet"]["depth"]
        self.complexity = settings["model"]["UNet"]["complexity"]
        self.num_convs = settings["model"]["UNet"]["num_convs"]

        # init comm-specific part
        self.num_comm_fmaps = settings["model"]["comm"]["num_comm_fmaps"]
        if self.num_comm_fmaps == 0:
            self.comm = False
        else:
            self.comm = settings["model"]["comm"]["comm"]
        self.subdom_dist = settings["data"]["subdomains_dist"]
        self.nx, self.ny = settings["data"]["subdomains_dist"]
        self.bilinear = bilinear

        self.communicator_type = communicator_type
        if not comm_network_but_no_communication is None:
            self.comm_network_but_no_communication = (not settings["comm"]["exchange_fmaps"])
        else:
            self.comm_network_but_no_communication = comm_network_but_no_communication
        self.communication_network_def = communication_network_def

        self.init_encoders()
        self.init_decoders()
        
        if self.comm:
            self.communication_network = self.communication_network_def(in_channels=self.num_comm_fmaps, out_channels=self.num_comm_fmaps, 
                                                                        dropout_rate=self.dropout_rate, kernel_size=self.kernel_size, padding=self.padding).to(devices[0])

    def init_encoders(self):
        encoder = Encoder(n_channels=self.n_channels, depth=self.depth, complexity=self.complexity,
                          dropout_rate=self.dropout_rate, kernel_size=self.kernel_size, num_convs=self.num_convs)
        self.encoders = nn.ModuleList([copy.deepcopy(encoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])

    def init_decoders(self):
        decoder = Decoder(n_channels=self.n_channels, depth=self.depth, n_classes=self.n_classes,
                          complexity=self.complexity, dropout_rate=self.dropout_rate, kernel_size=self.kernel_size, num_convs=self.num_convs)
        self.decoders = nn.ModuleList([copy.deepcopy(decoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])

    def _select_device(self, index):
        return self.devices[index % len(self.devices)]
    
    def _synchronize_all_devices(self):
        for device in self.devices:
            torch.cuda.synchronize(device=device)
        
    def _get_list_index(self, i, j):
        return i * self.ny + j
    
    def _get_grid_index(self, index):
        return index // self.ny, index % self.ny
    
    def concatenate_tensors(self, tensors):
        concatenated_tensors = []
        for i in range(self.nx):
            column_tensors = []
            for j in range(self.ny):
                index = self._get_list_index(i, j)
                column_tensors.append(tensors[index].to(self._select_device(0)))
            concatenated_row = torch.cat(column_tensors, dim=2)
            concatenated_tensors.append(concatenated_row)

        return torch.cat(concatenated_tensors, dim=3)

    def _split_concatenated_tensor(self, concatenated_tensor):
        subdomain_tensors = []
        subdomain_height = concatenated_tensor.shape[3] // self.nx
        subdomain_width = concatenated_tensor.shape[2] // self.ny

        for i in range(self.nx):
            for j in range(self.ny):
                subdomain = concatenated_tensor[:, :, j * subdomain_height: (j + 1) * subdomain_height,
                            i * subdomain_width: (i + 1) * subdomain_width]
                subdomain_tensors.append(subdomain)

        return subdomain_tensors
        
    def forward(self, input_image_list):
        assert len(input_image_list) == self.nx * self.ny, f"Number of input images must match the device grid size (nx x ny):= ({self.nx}x{self.ny}) but is {len(input_image_list)}."
        
        # Send to correct device and pass through encoder
        input_images_on_devices = [input_image.to(self._select_device(index)) for index, input_image in enumerate(input_image_list)]
        outputs_encoders = [self.encoders[index](input_image) for index, input_image in enumerate(input_images_on_devices)]

        # Do the communication step. Replace the encoder outputs by the communication output feature maps
        inputs_decoders = [[x.clone() for x in y] for y in outputs_encoders]

        if self.comm:
            if not self.comm_network_but_no_communication:
                communication_input = self.concatenate_tensors([output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders])
                communication_output = self.communication_network(communication_input)
                communication_output_split = self._split_concatenated_tensor(communication_output)
                
                for idx, output_communication in enumerate(communication_output_split):
                    inputs_decoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication
            
            elif self.comm_network_but_no_communication:        
                communication_inputs = [output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders]
                communication_outputs = [self.communication_network(comm_input.to(self.devices[0])) for comm_input in communication_inputs]
                
                for idx, output_communication in enumerate(communication_outputs):
                    inputs_decoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication.to(self._select_device(idx))
                    
        # Do the decoding step
        outputs_decoders = [self.decoders[index](output_encoder) for index, output_encoder in enumerate(inputs_decoders)]
        prediction = self.concatenate_tensors(outputs_decoders)
               
        return prediction

    def save_weights(self, save_path):
        state_dict = {
            'encoder_state_dict': [self.encoders[0].state_dict()],
            'decoder_state_dict': [self.decoders[0].state_dict()]
        }
        if self.comm:
            state_dict['communication_network_state_dict'] = self.communication_network.state_dict()
        torch.save(state_dict, save_path)

    def load_weights(self, load_path, device="cuda:0"):
        checkpoint = torch.load(load_path, map_location=device)
        encoder_state = checkpoint['encoder_state_dict'][0]
        decoder_state = checkpoint['decoder_state_dict'][0]

        for encoder in self.encoders:
            encoder.load_state_dict(encoder_state)
        for decoder in self.decoders:
            decoder.load_state_dict(decoder_state)
        if self.comm and 'communication_network_state_dict' in checkpoint:
            self.communication_network.load_state_dict(checkpoint['communication_network_state_dict'])
        else:
            print("No communication network found in dataset / no comm. network found")
            
    def parameters(self):
              
        if self.comm:
            parameters = list(self.encoders[0].parameters()) + list(self.decoders[0].parameters()) + list(self.communication_network.parameters()) 
        else:
            parameters = list(self.encoders[0].parameters()) + list(self.decoders[0].parameters())
            
        return parameters