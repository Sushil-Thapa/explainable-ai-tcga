import numpy as np


def predict(model, label_encoder, data):
    test_range = range(10)
    
    (X_train, X_test, y_train, y_test, label_encoder) = data

    
    # ## Evaluating with Validation set
    predicted_valid_labels = np.argmax(model.predict(X_test), axis=1)
    valid_labels = np.argmax(y_test, axis=1)
    
    print("Predicted labels: ", predicted_valid_labels[test_range])
    print("True labels: ", valid_labels[test_range])

    real = label_encoder.inverse_transform(valid_labels[test_range])
    predicts = label_encoder.inverse_transform(predicted_valid_labels[test_range])
    print("real:preds\n",{real[i]:predicts[i] for i in test_range})

    # Visualization of Confusion Matrix
    # import seaborn as sns

    # cm = metrics.confusion_matrix(valid_labels, predicted_valid_labels)
    # # print(cm)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plt.figure(figsize=(20,20))
    # sns.heatmap(cm_normalized, annot=True, fmt=".4f", linewidths=.5, square = True, cmap = 'summer')
    # plt.xlabel('Predicted Values', size=20)
    # plt.ylabel('Actual Values', size=20)

    # ticks = np.arange(len(set(valid_labels)))
    # tick_marks = ['Adrenal', 'Bile', 'Bladder', 'Bone', 'Brain', 'Breast', 'Cervix',
    #        'Colon', 'Esophagus', 'Fallopian', 'Head', 'Kidney', 'Leukemia',
    #        'Liver', 'Lung', 'Lymph', 'Mediastinum', 'Nervous', 'Ocular',
    #        'Ovarian', 'Pancreas', 'Pelvis', 'Peritoneum', 'Pleura',
    #        'Prostate', 'Rectum', 'Sarcoma', 'Skin', 'Stomach', 'Testis',
    #        'Thymus', 'Thyroid', 'Uterus', 'none']

    # plt.xticks(ticks+0.5 ,tick_marks, rotation=90, size=12) #add 0.5 to ticks to position it at center
    # plt.yticks(ticks+0.5 ,tick_marks, rotation=0, size=12)
    # # all_sample_title = 'Accuracy Score: {:.4f}'.format(93) # hardcoded this from training logs for now :D
    # # plt.title(all_sample_title, size = 30)
    # plt.show()