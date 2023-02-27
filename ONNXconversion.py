import torch
import onnx
import onnxruntime as ort
from torch import nn 



class CNNLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        
        # input "image" size here is torch.Size([25, 1, 30, 6])
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (5, 3), stride=(2, 1), padding=(4, 2))
        # height: input_size-filter_size +2(padding)/stride + 1 = 30-5+2(4)/2+1=17
        # width: 6-3+2*2/1 + 1 = 8
        # torch.Size([25, 8, 17, 8])
        self.batchnorm1 = nn.BatchNorm2d(8)
        # torch.Size([25, 8, 17, 8])
        # xcnnput_channel:8, batch(8)
        self.relu = nn.ReLU()
        # torch.Size([25, 8, 17, 8])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4,2))
        # torch.Size([25, 8, 4, 4])
        #input_size=28/2=14
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size = (5, 1), stride=(2, 1), padding=(1, 1))
        # same_padding: (5-1)/2=2:padding_size. 
        # torch.Size([25, 32, 1, 6])
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=1)
        self.fc1 = nn.Linear(in_features=192, out_features=96)
        # Nx3 * 3xO = NxO
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 =nn.Linear(in_features=96, out_features=3)
        self.fc3 =nn.Linear(in_features=num_sensors, out_features=3)

        self.linear = nn.Linear(
            in_features=self.hidden_units, 
            out_features=3
            )

    def forward(self, x):
        
        batch_size = x.shape[0]
        
        # The CNN part:
        xcnn = x
        xcnn = xcnn.reshape(batch_size,1,x.shape[1],x.shape[2])
        xcnn = self.cnn1(xcnn)
        xcnn =self.batchnorm1(xcnn)
        xcnn =self.relu(xcnn)
        xcnn =self.maxpool1(xcnn)
        xcnn =self.cnn2(xcnn)
        xcnn =self.batchnorm2(xcnn)
        xcnn =self.relu(xcnn)
        xcnn =self.maxpool2(xcnn)
        xcnn = torch.flatten(xcnn,1)
        xcnn =self.fc1(xcnn)
        xcnn =self.relu(xcnn)
        xcnn =self.dropout(xcnn)
        xcnn =self.fc2(xcnn)
        
        # The LSTM part
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()  # torch.Size([1, 25, 16])
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        xlstm = self.linear(hn[0])  # First dim of Hn is num_layers, which is set to 3 above.

        
        # concat
        out = torch.cat((xcnn,xlstm), 1)
        out  = self.fc3(out)
        
        return out

num_hidden_units = 16
num_sensors = 6

model = CNNLSTM(num_sensors=num_sensors, hidden_units=num_hidden_units)

# Load PyTorch model
model.load_state_dict(torch.load('models\CNNLSTM.pth'))

# Set model to evaluation mode
model.eval()

# Create input tensor
dummy_input = torch.randn(1, 30, 6)

# Export PyTorch model to ONNX
input_names = ['emg_elbow','emg_shfe','emg_shaa','pos_elbow','pos_shfe','pos_shaa']
output_names = ["predicted_positions"]
torch.onnx.export(model, dummy_input, "models\converted_cnnlstm.onnx", verbose=True, input_names=input_names, output_names=output_names)

# Load ONNX model
onnx_model = onnx.load("models\converted_cnnlstm.onnx")

# Create Inference session using ONNX model
ort_session = ort.InferenceSession("models\converted_cnnlstm.onnx")

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Create input tensor
input_tensor = dummy_input.numpy()

# Run inference
outputs = ort_session.run([output_name], {input_name: input_tensor})

# Print outputs
print(outputs)
