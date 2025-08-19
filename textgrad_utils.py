import numpy as np
import torch
import textgrad as tg
from tqdm import tqdm
import concurrent

def serialize_data(batch_x, batch_y=None):

    data_str_list = []
    
    for i, features in enumerate(batch_x):
        # Extract feature values
        sepal_length = features["sepal length (cm)"]
        sepal_width = features["sepal width (cm)"]
        petal_length = features["petal length (cm)"]
        petal_width = features["petal width (cm)"]
        
        # Determine label
        if batch_y is not None:
            label = batch_y[i]
            data_str = (
                f"The sample has a sepal length of {sepal_length} cm, "
                f"sepal width of {sepal_width} cm, "
                f"petal length of {petal_length} cm, "
                f"and petal width of {petal_width} cm. The label is {label}."
            )
        else:
            data_str = (
                f"The sample has a sepal length of {sepal_length} cm, "
                f"sepal width of {sepal_width} cm, "
                f"petal length of {petal_length} cm, "
                f"and petal width of {petal_width} cm. The label is ?"
            )
        
        data_str_list.append(data_str)
    
    return data_str_list

def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)

    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)
    
def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)

    