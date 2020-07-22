import numpy as np
import lime
import lime.lime_tabular
import time

# we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles

def explain(model, data):
    (X_train, X_test, y_train, y_test, label_encoder, _) = data
    start = time.time()
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                    feature_names=df.columns, 
                                                    class_names=label_encoder.classes_)
    print("Explainer Elapsed time:", time.time() - start)


    for _ in range(2):
        start = time.time()
        ith = np.random.randint(0, X_test.shape[0])
        print(ith,"th test sample")
        exp = explainer.explain_instance(X_test[ith], model.predict_proba, num_features=20, top_labels=5)
        
        # exp.show_in_notebook(show_table=True, show_all=True)
        
        for i in exp.available_labels():
            print(i,"/len({exp.available_labels())th class: ", label_encoder.classes_[i])
        #     display(pd.DataFrame(exp.as_list(label=i)))
            print(exp.as_list(label=i))

        print("Iteration Elapsed time:", time.time() - start)

    # For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.


    start = time.time()
    i = np.random.randint(0, X_test.shape[0])
    print(i)
    exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=10, top_labels=5)
    print(time.time() - start)

    exp.show_in_notebook(show_table=True, show_all=False)

    # feature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
    # Float point number on the horizontal bars represent the relative importance of these features.

    exp.show_in_notebook(show_table=True, show_all=True)

    #for easier analysis and further processing
    for i in exp.available_labels():
        print(label_encoder.classes_[i])
    #     display(pd.DataFrame(exp.as_list(label=i)))
        print(exp.as_list(label=i))

