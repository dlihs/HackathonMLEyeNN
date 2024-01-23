from preprocessing import Preprocessor
from model import CNNModel

if __name__ == "__main__":
    path_to_file = '2_left.jpg'
    normalized_image = Preprocessor(path_to_file)
    
    if normalized_image is not None:
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0).unsqueeze(0).float()
        model = CNNModel()
        model.eval() 
    
        # Make a prediction
        with torch.no_grad():
            result = model(image_tensor)
        print(result)
    else:
        print("Image preprocessing failed.")