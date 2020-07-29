import numpy as np
import lime
import lime.lime_tabular
import time

# we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles

def explain(model, data, n_samples=1, submodular_pick= False):
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


    if submodular_pick == False:
        for _ in range(n_samples):
            start = time.time()
            print("Starting explaining the instance...")
            ith = np.random.randint(0, X_test.shape[0])
            ith = 774
            print(ith,"th test sample")

            # For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.
            exp = explainer.explain_instance(X_test[ith], model.predict_proba, num_features=num_features, top_labels=top_labels)
            
            # exp.show_in_notebook(show_table=True, show_all=False)
            # feature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
            # Float point number on the horizontal bars represent the relative importance of these features.

            exp_avilable_labels = exp.available_labels()
            print("Number of labels to analyze",len(exp_avilable_labels))

            html_out = f"out/lime/{ith}.html"
            print("Saving explainations to file",html_out)
            exp.save_to_file(html_out)
            print("Error R2 Score:",exp.score) # 0-1 worse-better
            
            for i in exp_avilable_labels:
                print(f"\n\n{ith}sample, {i}th class explaination: {label_encoder.classes_[i]}")
            #     display(pd.DataFrame(exp.as_list(label=i)))
                print(exp.as_list(label=i))

            print("Iteration complete, Elapsed time:", time.time() - start)

    else:
        print("Preparing submodular engines...")
        num_exps_desired = 5
        sample_size = 10

        import warnings
        import pandas as pd

        from lime import submodular_pick

        start = time.time()
        sp_obj = submodular_pick.SubmodularPick(explainer, X_train, model.predict_proba, sample_size = sample_size, num_features=num_features,num_exps_desired=num_exps_desired)
        print("Submodular pick complete. Elapsed time:", time.time() - start)

        df=pd.DataFrame({})
        for this_label in range(top_labels):
            start = time.time()
            dfl=[]
            for i,exp in enumerate(sp_obj.sp_explanations):
                l=exp.as_list(label=this_label)  # this label should be for looped or for ranged.
                l.append(("exp number",i))
                dfl.append(dict(l))
            # dftest=pd.DataFrame(dfl)
            df=df.append(pd.DataFrame(dfl,index=[label_encoder.classes_[this_label] for i in range(len(sp_obj.sp_explanations))]))
            print(df)
            print("Iteration complete, Elapsed time:", time.time() - start)

        df.to_csv("out/lime/sp_out.csv")



    



