import pitchDetectionNetTraining as trainFunc

# Choose folder containing desired dataset
# and type in the name of the dataset
folder = 'TrainingDatasets/Guitar_Trump/'
dataset = 'GTSetF2048_H512_Take_3_Size_18722_4AscOnly.h5'
# Get training data and labels into RAM
train, labels = trainFunc.get_2_inst_data(folder, dataset)
# Run K=5-Fold test, (model, saveFileName, train, labels, dataset, epochs)
trainFunc.log5KFoldTest_two_inst(trainFunc.shallow_2_inst_model_both_tallThin_8feat(), 'shallow_2_inst_model_both_tallThin_8feat', train, labels, dataset, epochs = 20)

