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

import re
def parse_accuracy_from_tag(text):
    match = re.search(r"<Accuracy>\s*(\d(?:\.\d+)?)\s*</Accuracy>", text)
    if match:
        return int(float(match.group(1)))  # safe for both int/float style values
    else:
        return 0  # fallback, or raise error

    # 修改這行：
    return parse_accuracy_from_tag(eval_output_variable.value)

def eval_sample(item, eval_fn, model, prompt_with_data):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item

    x = serialize_data([x])

    x = f"{prompt_with_data.value}\n" + "\n" + x[0]


    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    
    response = model(x)

    comparison = f"prediction ={response} ground truth = {y} features values = {x.value}"

    comparison = tg.Variable(comparison, requires_grad=False, role_description="evaluation of the prediction against the ground truth")

    eval_output_variable = eval_fn(comparison)

    #print("eval_output_variable: ", type(parse_accuracy_from_tag(eval_output_variable.value)))

    return parse_accuracy_from_tag(eval_output_variable.value)
    #return int(eval_output_variable.value)

    
def eval_dataset(test_loader, eval_fn, model, system_prompt ,max_samples: int=None):

    for batch_x, batch_y in test_loader:
        print("batch: ", batch_y)

        batch_x = np.array(batch_x)  # Convert to numpy array if not already
        batch_y = np.array(batch_y)  # Convert to numpy array if not already
        # 2. randomly sample 10 data points
        sample_size = min(10, len(batch_x))  # Ensure we don't sample more than available
        indices = np.random.choice(len(batch_x), sample_size, replace=False)

        sampled_batch_x = batch_x[indices]
        sampled_batch_y = batch_y[indices]

        # Extract the rest of the data (excluding sampled)
        remaining_batch_x = np.delete(batch_x, indices, axis=0)
        remaining_batch_y = np.delete(batch_y, indices, axis=0)

        # 3. serialize the data
        serialized_sample_data = serialize_data(sampled_batch_x, sampled_batch_y)

        # 4. concatenate sample and data them with prompt
        prompt_with_data = f"{system_prompt.value}\n" + "\n".join(serialized_sample_data) + "\n" + "what is the answer below? Just return the species name of flower don't give me anything else"
        prompt_with_data = tg.Variable(prompt_with_data, requires_grad=False, role_description="query to the language model with sample data")

        
        if max_samples is None:
            max_samples = len(remaining_batch_x)
        accuracy_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for _, sample in enumerate(zip(remaining_batch_x, remaining_batch_y)):
                
                future = executor.submit(eval_sample, sample, eval_fn, model, prompt_with_data)
                futures.append(future)
                if len(futures) >= max_samples:
                    break
            tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
            for future in tqdm_loader:
                acc_item = future.result()
                accuracy_list.append(acc_item)
                tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")

        print("accuracy_list: ", accuracy_list)
    return accuracy_list 


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model, system_prompt))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    #results["validation_acc"].append(val_performance)

    return system_prompt, val_performance

    