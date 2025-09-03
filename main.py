from torch.utils.data import DataLoader
from textgrad_utils import serialize_data, eval_sample, eval_dataset, run_validation_revert
from dataloader import IrisDataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import textgrad as tg
from textgrad.loss import TextLoss, MultiFieldTokenParsedEvaluation
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
load_dotenv(override=True)


# Load Iris dataset
iris = load_iris(as_frame=True)
X = iris.data.to_dict(orient="records")  # Features as a list of dictionaries
y = iris.target
y = pd.Series(y).map(lambda i: iris.target_names[i])
print(y)

# Split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% val, 20% test

# Create Dataset instances for each split
train_dataset = IrisDataset(X_train, y_train)
val_dataset = IrisDataset(X_val, y_val)
test_dataset = IrisDataset(X_test, y_test)

def collate_fn(batch):
    batch_x = [item["input"] for item in batch]
    batch_y = [item["target"] for item in batch]
    return batch_x, batch_y

# Create DataLoaders for each split
train_batch_size = 20  # Define your batch size
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn)

# Initial prompt
system_prompt = "Predict the species of the iris flower based on its sepal and petal dimensions."
system_prompt = tg.Variable(system_prompt, 
                        requires_grad=True, 
                        role_description="system prompt to the language model")



# 2. set up model
llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo-0125")
llm_api_eval = tg.get_engine(engine_name="gpt-4o")

gpt3_model = tg.BlackboxLLM(llm_api_test)
gpt4_model = tg.BlackboxLLM(llm_api_eval)

tg.set_backward_engine(llm_api_eval, override=True)

optimizer_system_prompt = """
You're optimizing the system prompt of a language model classifier for the iris flower dataset.
Given the classification performance feedback, return an improved version of the current system prompt.
Only output the new prompt within <new_prompt> and </new_prompt> tags.

Example:
<new_prompt> Predict iris flower species using its petal and sepal measurements. </new_prompt>
"""


optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt], optimizer_system_prompt=optimizer_system_prompt, new_variable_tags=["<new_prompt>", "</new_prompt>"])

role_descriptions = [
    "Question for the task",
    "Ground truth answer",
    "Prediction from the language model"
]
#evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <Accuracy> </Accuracy> tags. e.g.<Accuracy> 0 </Accuracy> or <Accuracy> 1 </Accuracy> and also give me your percentage's confident between 0% to 100% for this answer and put it in <Confident></Confident>, also put the feature value with its name in <Features></Features> tags format like this sepal length: ? cm, sepal width: ? cm, petal length: ? cm, petal width: ? cm, also put your natural language feedback in <FEEDBACK> </FEEDBACK> tags. "
evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <Accuracy> </Accuracy> tags. e.g.<Accuracy> 0 </Accuracy> or <Accuracy> 1 </Accuracy> "

#evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer?"
eval_instruction = tg.Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")

eval_fn = TextLoss(eval_instruction, engine=llm_api_eval)



def pick_samples_from_train_data(data, labels, sample_size=10):

    # we have to change to KNN sampling later
    indices = np.random.choice(len(data), sample_size, replace=False)
    sampled_data = data[indices]
    sampled_labels = labels[indices]
    return {"data": sampled_data, "label": sampled_labels}

samples = pick_samples_from_train_data(np.array(X_train), np.array(y_train), sample_size=8)


results = {"test_acc": [], "prompt": [], "validation_acc": []}
results["test_acc"].append(eval_dataset(test_loader, eval_fn, gpt3_model, system_prompt, samples=samples))
results["validation_acc"].append(eval_dataset(val_loader, eval_fn, gpt3_model, system_prompt, samples=samples))
results["prompt"].append(system_prompt.get_value())


# 1. Initial prediction
for epoch in range(1):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):

        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []
        # 1. import data
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

        #print("prompt with data: ", prompt_with_data.value)
        
        for (x, y) in zip(remaining_batch_x, remaining_batch_y):

            print(type(x))
            x = serialize_data([x])
            print("x after serialize: ", x)

            x = f"{prompt_with_data.value}\n" + "\n" + x[0]

            print("x after adding prompt: ", x)

            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")


            response = gpt3_model(x)

            print("response: ", response, "ground truth: ", y)

            #comparison = f"prediction ={response} ground truth = {y} features values = {x.value}"
            comparison = f"prediction ={response} ground truth = {y}"

            comparison = tg.Variable(comparison, requires_grad=False, role_description="evaluation of the prediction against the ground truth")
            eval_output_variable = eval_fn(comparison)

            print("eval_output_variable: ", eval_output_variable)

            losses.append(eval_output_variable)

        print("losses: ", losses)
        total_loss = tg.sum(losses)
        print(total_loss)

        #----- fix below -----

        total_loss.backward()

        optimizer.step()

        system_prompt, val_performance = run_validation_revert(system_prompt, results, gpt3_model, eval_fn, val_loader, samples)
        
        results["validation_acc"].append(val_performance)
        test_acc = eval_dataset(test_loader, eval_fn, gpt3_model, system_prompt, samples=samples)
        results["test_acc"].append(test_acc)
        results["prompt"].append(system_prompt.get_value())

print("this is the final result:\n")
print(results)
test_acc_scores = [np.mean(batch) for batch in results['test_acc']]
valid_acc_scores = [np.mean(batch) for batch in results['validation_acc']]

results['test_acc'] = test_acc_scores
results['validation_acc'] = valid_acc_scores

print(results)

