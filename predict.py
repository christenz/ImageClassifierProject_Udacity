#imports

import predict_functions as pf

def main():
    """main function of the predicting program"""
    #get user input
    in_args = pf.get_input_arguments()
    
    #set device:
    if in_args.GPU == True:
        device = "cuda"
    else:
        device = "cpu"
    
    print(in_args)
    #load model
    model = pf.load_checkpoint(in_args.checkpoint, device)
    
    #preprocess image
    np_image = pf.process_image(in_args.image_path)
    
    #predict image
    classes, probabilities = pf.predict_image(np_image, model, top_k = in_args.top_k)
    
    #get label mapping
    cat_to_name = pf.get_label_mapping("ImageClassifier/cat_to_name.json")

    #create a dictionary which shows the names of the predicted classes & their likelihoods
    prediction_dict = pf.get_class_names(classes, probabilities, cat_to_name)
    print(prediction_dict)
main()