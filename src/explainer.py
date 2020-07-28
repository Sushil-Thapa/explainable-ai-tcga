import numpy as np
import lime
import lime.lime_tabular
import time

# we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles

def explain(model, data):
    (X_train, X_test, y_train, y_test, feature_names, label_encoder) = data
    start = time.time()
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                    feature_names=feature_names, 
                                                    class_names=label_encoder.classes_)
    print("Explainer Elapsed time:", time.time() - start)


    for _ in range(2):
        start = time.time()
        ith = np.random.randint(0, X_test.shape[0])
        print(ith,"th test sample")

        # For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.
        exp = explainer.explain_instance(X_test[ith], model.predict_proba, num_features=20, top_labels=5)
        
        # exp.show_in_notebook(show_table=True, show_all=False)
        # feature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
        # Float point number on the horizontal bars represent the relative importance of these features.

        exp_avilable_labels = exp.available_labels()
        for i in exp_avilable_labels:
            print(f"{i}/{len(exp_avilable_labels)}th class: {label_encoder.classes_[i]}")
        #     display(pd.DataFrame(exp.as_list(label=i)))
            print(exp.as_list(label=i))

        print("Iteration Elapsed time:", time.time() - start)

    



