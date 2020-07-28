import numpy as np
import lime
import lime.lime_tabular
import time

# we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles

def explain(model, data, n_samples=1):
    print("Starting Explainer module...")
    num_features = 10
    top_labels = 3

    (X_train, X_test, y_train, y_test, feature_names, label_encoder) = data

    start = time.time()
    print("Preparing tabular explainer...")
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                    feature_names=feature_names, 
                                                    class_names=label_encoder.classes_)
    print("Explainer preparation complete. Elapsed time:", time.time() - start)


    for _ in range(n_samples):
        start = time.time()
        print("Starting explaining the instance...")
        ith = np.random.randint(0, X_test.shape[0])
        # ith = 774
        print(ith,"th test sample")

        # For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.
        exp = explainer.explain_instance(X_test[ith], model.predict_proba, num_features=num_features, top_labels=top_labels)
        
        # exp.show_in_notebook(show_table=True, show_all=False)
        # feature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
        # Float point number on the horizontal bars represent the relative importance of these features.

        exp_avilable_labels = exp.available_labels()

        html_out = f"out/lime/{ith}.html"
        print("Saving explainations to file",html_out)
        exp.save_to_file(html_out)
        print("Error R2 Score:",exp.score) # 0-1 worse-better
        
        for i in exp_avilable_labels:
            print(f"\n\n{ith}sample, {i}/{len(exp_avilable_labels)}th class explainations: {label_encoder.classes_[i]}")
        #     display(pd.DataFrame(exp.as_list(label=i)))
            print(exp.as_list(label=i))

        print("Iteration Elapsed time:", time.time() - start)

    



