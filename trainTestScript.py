import pitchDetectionNetTraining as trainFunc

folder = 'TrainingDatasets/Guitar_Trump/'
dataset = 'GTSetF2048_H512_Take_3_Size_18722_4AscOnly.h5'
train, labels = trainFunc.get_2_inst_data(folder, dataset)
trainFunc.log5KFoldTest_two_inst(trainFunc.shallow_2_inst_model_both_tallThin_8feat(), 'shallow_2_inst_model_both_tallThin_8feat', train, labels, dataset, epochs = 20)  
## Not working atm 
'''
dataset = 'GTSetF1024_H512_Take_3_Size_37445_4AscOnly.h5'
train, labels = trainFunc.get_2_inst_data(folder, dataset)
trainFunc.log5KFoldTest_two_inst(trainFunc.shallow_2_inst_model(), 'shallow_2_inst_model', train, labels, dataset, epochs = 20) '''

