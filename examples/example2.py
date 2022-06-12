from tm_eval import *

if __name__=="__main__":
    # start configure
    # load a dictionary with document id as key and its term list split by ',' as value.
    input_file = "datasets/covid19_symptoms.pickle"
    output_folder = "outputs"
    model_name = "symptom"
    start=2
    end=20
    # end configure
    # run and explore

    list_results = explore_topic_model_metrics(input_file=input_file, output_folder=output_folder,
                                                  model_name=model_name,start=start,end=end)
    # summarize results
    show_topic_model_metric_change(list_results,save=True,save_path=f"{output_folder}/metrics.csv")

    # plot metric changes
    plot_tm_metric_change(csv_path=f"{output_folder}/metrics.csv",save=True,save_folder=output_folder)
