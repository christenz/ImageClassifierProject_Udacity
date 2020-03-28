#imports
import train_functions as tf

def main():
    """main function of the training program"""
    #get user input
    in_args = tf.get_input_arguments()
    print(in_args)
    #set device:
    
    if in_args.GPU == "True":
        device = "cuda"
    else:
        device = "cpu"
    
    #create data loaders
    train_loader, valid_loader, test_loader, train_datasets = tf.get_data_loader(in_args.data_dir)
        
    #get model
    model = tf.get_model(in_args.arg, in_args.hidden_units_1, in_args.hidden_units_2)
    
    #train model
    trained_model = tf.train_model(model, train_loader=train_loader, validation_loader=valid_loader, device = device, learning_rate= in_args.learning_rate, epochs = in_args.epochs)
    
    #test model
    test_accuracy, test_loss = tf.test_model(trained_model, test_loader, device = device)
    print("test_accuracy, test_loss:")
    print(test_accuracy, test_loss)
    
    #save trained model in checkpoint
    tf.save_dictionary(trained_model, train_datasets, "checkpoint.pth")

main()