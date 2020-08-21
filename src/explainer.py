import os

from src.settings import SEED_VALUE, lime_out_path, splime_out_path
# os.environ['PYTHONHASHSEED']=str(SEED_VALUE)

import time
import numpy as np
import lime
import lime.lime_tabular
import random


# we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles

def explain(model, data, explain_data, srt_idx=0, n_samples=3, submodular_pick= False):
    print("Starting Explainer module...")
    num_features = 10  # maximum number of features present in the explainations
    top_labels = 1  # number of max probable classes to consider

    (X_train, X_test, y_train, y_test, feature_names, label_encoder, fpkm_data) = data

    print('Predicted label: ', np.argmax(model.predict(explain_data), axis=1))
    
    start = time.time()
    print("Preparing tabular explainer...")
    print("\n\nComment:  Verified reproducible results.")
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                    feature_names=feature_names, 
                                                    class_names=label_encoder.classes_, random_state=SEED_VALUE)
    print("Explainer preparation complete. Elapsed time:", time.time() - start)


    if submodular_pick:
        print("Preparing submodular-pick engines...")
        num_exps_desired = 5  # number of exp objects returned.
        sample_size = 5 # number of instances to explain 

        import warnings
        import pandas as pd

        from lime import submodular_pick

        start = time.time()
        sp_obj = submodular_pick.SubmodularPick(explainer, X_train, model.predict_proba, top_labels = top_labels, \
            sample_size = sample_size, num_features=num_features,num_exps_desired=num_exps_desired)
        print("Submodular pick complete. Elapsed time:", time.time() - start)

        df=pd.DataFrame({})
        for this_label in range(top_labels):
            start = time.time()
            dfl=[]
            for i,exp in enumerate(sp_obj.sp_explanations):
                print("R2 Score:",exp.score) # 0-1 worse-better
                
                exp_avilable_labels = exp.available_labels()

                #TODO save html
                html_out = f"out/lime/sp/{this_label}_{i}.html"
                print("Saving explainations to file",html_out)
                exp.save_to_file(html_out)

                l=exp.as_list(label=exp_avilable_labels[this_label])  # for looped instead of for looped(in tutorial) as not all are in available labels
                l.append(("exp number",i))
                dfl.append(dict(l))
            # dftest=pd.DataFrame(dfl)
            df=df.append(pd.DataFrame(dfl,index=[label_encoder.classes_[exp_avilable_labels[this_label]] for i in range(len(sp_obj.sp_explanations))]))
            print(df)
            print("Iteration complete, Elapsed time:", time.time() - start)

        df.to_csv(f"{splime_out_path}/sp_out.csv")



    else:
        for n in range(n_samples):  #?????
            start = time.time()
            print("Starting explaining the instance...",n)
            #ith = np.random.randint(0, X_test.shape[0])
            #ith = 774
            for ith, d in enumerate(explain_data):
                print(f"\n{srt_idx}_{ith}th sample \t Run: {n}")
                # For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.

                # random.seed(SEED_VALUE)
                # np.random.seed(SEED_VALUE)

                exp = explainer.explain_instance(d, model.predict_proba, num_features=num_features, \
                    top_labels=top_labels, num_samples=10000)
            
                # exp.show_in_notebook(show_table=True, show_all=False)
                # feature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
                # Float point number on the horizontal bars represent the relative importance of these features.

                exp_avilable_labels = exp.available_labels()
                print("Number of labels to analyze",len(exp_avilable_labels))

                html_out = f"{lime_out_path}/{n}_{srt_idx}_{ith}.html"
                print("Saving explainations to file",html_out)
                exp.save_to_file(html_out)
                print("Error R2 Score:",exp.score) # 0-1 worse-better
            
                for i in exp_avilable_labels:
                    print(f"\nsample {srt_idx+ith}: class {i}: {label_encoder.classes_[i]}")
                #     display(pd.DataFrame(exp.as_list(label=i)))
                    print(exp.as_list(label=i))

            print("Iteration complete, Elapsed time:", time.time() - start)


    



