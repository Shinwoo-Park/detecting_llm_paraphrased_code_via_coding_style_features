import random 
import time 
import argparse
import pickle 
import numpy as np
import json 
from sklearn.metrics import f1_score
import re 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)

# Customized K-Fold Cross Validation
def create_randomized_kfold_datasets(data, k, seed):
    if k <= 1:
        raise ValueError("K should be greater than 1.")
    if len(data) < k:
        raise ValueError("The length of data should be greater than K.")
    
    random.seed(seed)
    randomized_data = data[:]
    random.shuffle(randomized_data)
    
    partitions = []
    n = len(randomized_data)
    fold_size = n // k
    remainder = n % k 

    start_idx = 0
    for i in range(k):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + fold_size + extra
        partitions.append(randomized_data[start_idx:end_idx])
        start_idx = end_idx
    
    kfold_datasets = []
    for i in range(k):
        val_set = partitions[i]
        train_set = [item for j in range(k) if j != i for item in partitions[j]]
        kfold_datasets.append((train_set, val_set))
    
    return kfold_datasets

# Feature Extraction
def classify_naming(name):
    """Simplify naming convention classification."""
    if re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', name):
        return 'camelCase'
    elif re.match(r'^[a-z]+(?:_[a-z]+)+$', name):
        return 'snake_case'
    elif re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', name):
        return 'PascalCase'
    elif re.match(r'^[A-Z]+(?:_[A-Z]+)+$', name):
        return 'UPPER_SNAKE_CASE'
    else:
        return 'Other'

def extract_function_names(code, language):
    """Extract function names using regex for different languages."""
    if language == "py":
        pattern = r'^\s*def\s+(\w+)\s*\('
    elif language in ["c", "cpp", "java"]:
        pattern = r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:[\w<>\[\]\*&]+\s+)+(\w+)\s*\('
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)

def extract_variable_names(code, language):
    """Extract variable names using regex for different languages."""
    if language == "py":
        pattern = r'^\s*(\w+)\s*=\s*[^=]'
    elif language in ["c", "cpp", "java"]:
        pattern = r'^\s*(?:[\w<>\[\]\*&]+\s+)+(\w+)\s*(?:=|;|\[|\()'
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)

def extract_class_names(code, language):
    """Extract class names using regex for different languages."""
    if language == "py":
        pattern = r'^\s*class\s+(\w+)\s*(?:\(|:)?'
    elif language in ["c", "cpp", "java"]:
        pattern = r'^\s*(?:public|private|protected)?\s*(?:class|struct|interface)\s+(\w+)'
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)

def extract_constant_names(code, language):
    """Extract constant names using regex for different languages."""
    if language == "py":
        pattern = r'^\s*([A-Z][A-Z_0-9]*)\s*=\s*'
    elif language in ["c", "cpp"]:
        pattern = r'^\s*#define\s+([A-Z][A-Z_0-9]*)\b'
    elif language == "java":
        pattern = r'^\s*(?:public|private|protected)?\s*static\s+final\s+[\w<>\[\]]+\s+([A-Z][A-Z_0-9]*)\s*='
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)

def analyze_code(code, language):
    """
    Analyze a single code snippet and return its metrics.
    """
    lines = code.splitlines()
    total_lines = len(lines)

    # Initialize metrics
    metrics = {
        "function_naming_consistency": 0.0,
        "variable_naming_consistency": 0.0,
        "class_naming_consistency": 0.0,
        "constant_naming_consistency": 0.0,
        "indentation_consistency": 0.0,
        "avg_function_length": 0.0,
        "avg_nesting_depth": 0.0,
        "comment_ratio": 0.0,
        "avg_function_name_length": 0.0,
        "avg_variable_name_length": 0.0
    }

    # 1. Extract names
    function_names = extract_function_names(code, language)
    variable_names = extract_variable_names(code, language)
    class_names = extract_class_names(code, language)
    constant_names = extract_constant_names(code, language)

    # 2. Analyze naming conventions separately
    def calculate_naming_consistency(names):
        naming_counts = {
            "camelCase": 0,
            "snake_case": 0,
            "PascalCase": 0,
            "UPPER_SNAKE_CASE": 0,
            "Other": 0
        }
        for name in names:
            naming_style = classify_naming(name)
            if naming_style in naming_counts:
                naming_counts[naming_style] += 1
            else:
                naming_counts["Other"] += 1
        total_names = sum(naming_counts.values())
        if total_names > 0:
            most_common_style_count = max(naming_counts.values())
            return most_common_style_count / total_names
        else:
            return 0.0

    # Calculate naming consistency for each identifier type
    metrics["function_naming_consistency"] = calculate_naming_consistency(function_names)
    metrics["variable_naming_consistency"] = calculate_naming_consistency(variable_names)
    metrics["class_naming_consistency"] = calculate_naming_consistency(class_names)
    metrics["constant_naming_consistency"] = calculate_naming_consistency(constant_names)

    # Calculate average name lengths
    metrics["avg_function_name_length"] = sum(len(name) for name in function_names) / len(function_names) if function_names else 0.0
    metrics["avg_variable_name_length"] = sum(len(name) for name in variable_names) / len(variable_names) if variable_names else 0.0

    # 3. Analyze indentation consistency (continuous value)
    def calculate_indentation_consistency(lines):
        indent_unit_counts = {}
        total_indented_lines = 0
        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line or stripped_line.startswith(('#', '//', '/*', '*')):
                continue
            indent = line[:len(line)-len(stripped_line)]
            if indent:
                total_indented_lines += 1
                # Replace tabs with spaces (assuming a tab size of 4 spaces)
                indent = indent.replace('\t', '    ')
                indent_length = len(indent)
                if indent_length in indent_unit_counts:
                    indent_unit_counts[indent_length] += 1
                else:
                    indent_unit_counts[indent_length] = 1

        if total_indented_lines == 0:
            return 1.0  # No indentation used, considered consistent

        most_common_indent_count = max(indent_unit_counts.values())
        consistency = most_common_indent_count / total_indented_lines

        return consistency

    metrics["indentation_consistency"] = calculate_indentation_consistency(lines)

    # 4. Analyze function lengths (applies to all languages)
    function_lengths = []
    function_pattern = {
        "py": r'^\s*def\s+\w+\s*\(.*\):',
        "c": r'^\s*(?:[\w<>\[\]\*&]+\s+)+\w+\s*\(.*\)\s*\{',
        "cpp": r'^\s*(?:[\w<>\[\]\*&]+\s+)+\w+\s*\(.*\)\s*\{',
        "java": r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:[\w<>\[\]\.&]+\s+)+\w+\s*\(.*\)\s*\{'
    }.get(language)

    if function_pattern:
        function_starts = [i for i, line in enumerate(lines) if re.match(function_pattern, line)]
        for start_line in function_starts:
            length = 0
            nesting_level = 0
            i = start_line
            while i < len(lines):
                line = lines[i]
                stripped_line = line.strip()
                if language == "py":
                    current_indent = len(line) - len(line.lstrip())
                    start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                    if i > start_line and stripped_line and (len(line) - len(line.lstrip())) <= start_indent:
                        break
                else:
                    nesting_level += line.count('{') - line.count('}')
                    if i > start_line and nesting_level <= 0:
                        break
                length += 1
                i += 1
            function_lengths.append(length)
        metrics["avg_function_length"] = sum(function_lengths) / len(function_lengths) if function_lengths else 0.0

    # 5. Analyze nesting depth (applies to all languages)
    nesting_depths = []
    if language == "py":
        indent_levels = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue
            current_indent = len(line) - len(line.lstrip())
            while indent_levels and current_indent < indent_levels[-1]:
                indent_levels.pop()
            if indent_levels and current_indent == indent_levels[-1]:
                pass
            elif current_indent > (indent_levels[-1] if indent_levels else 0):
                indent_levels.append(current_indent)
            nesting_depths.append(len(indent_levels))
        metrics["avg_nesting_depth"] = sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0.0
    else:
        nesting_level = 0
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith(('//', '/*', '*')):
                continue
            nesting_level += line.count('{') - line.count('}')
            nesting_depths.append(max(nesting_level, 0))
        metrics["avg_nesting_depth"] = sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0.0

    # 6. Calculate comment ratio (applies to all languages)
    comment_lines = 0
    code_lines = 0
    in_block_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if language == "py":
            if stripped_line.startswith('#'):
                comment_lines += 1
            elif re.match(r'(\'\'\'|\"\"\")', stripped_line):
                comment_lines += 1
                if stripped_line.count('\'\'\'') % 2 == 1 or stripped_line.count('\"\"\"') % 2 == 1:
                    in_block_comment = not in_block_comment
            elif in_block_comment:
                comment_lines += 1
            else:
                code_lines += 1
        else:
            if in_block_comment:
                comment_lines += 1
                if '*/' in stripped_line:
                    in_block_comment = False
            elif stripped_line.startswith('/*'):
                comment_lines += 1
                if '*/' not in stripped_line:
                    in_block_comment = True
            elif stripped_line.startswith('//'):
                comment_lines += 1
            else:
                code_lines += 1

    total_code_lines = code_lines + comment_lines
    metrics["comment_ratio"] = comment_lines / total_code_lines if total_code_lines > 0 else 0.0

    feature = []
    feature.append(metrics["function_naming_consistency"])
    feature.append(metrics["variable_naming_consistency"])
    feature.append(metrics["class_naming_consistency"])
    feature.append(metrics["constant_naming_consistency"])
    feature.append(metrics["indentation_consistency"])
    feature.append(metrics["avg_function_length"])
    feature.append(metrics["avg_nesting_depth"])
    feature.append(metrics["comment_ratio"])
    feature.append(metrics["avg_function_name_length"])
    feature.append(metrics["avg_variable_name_length"])

    return feature

# LPcodedec
def ml_feature_prediction(kfold_dataset_feature, ml_model, args): 

    fold_f1_score = []
    fold_time = []

    for fold, dataset in kfold_dataset_feature.items():

        print(f"Fold {fold + 1}...")

        start_time = time.time()

        train_dataset = dataset['train']
        test_dataset = dataset['test']

        train_feature = []
        test_feature = []
        train_label = []
        test_label = []

        for inst in train_dataset:
            feature = [] 
            feature.extend(analyze_code(inst['human_src'], args.lang))
            feature.extend(analyze_code(inst['llm_src'], args.lang))
            train_feature.append(feature)
            train_label.append(inst['label'])
        for inst in test_dataset:
            feature = [] 
            feature.extend(analyze_code(inst['human_src'], args.lang))
            feature.extend(analyze_code(inst['llm_src'], args.lang))
            test_feature.append(feature)
            test_label.append(inst['label'])

        scaler = StandardScaler()
        train_feature = scaler.fit_transform(train_feature)
        test_feature = scaler.transform(test_feature)

        if ml_model == 'MLPClassifier':
            model = MLPClassifier(random_state=args.seed)

        model.fit(train_feature, train_label)
        predictions = model.predict(test_feature)

        # Calculate metrics
        f1 = f1_score(test_label, predictions)
        fold_f1_score.append(f1)
        end_time = time.time()
        fold_time.append(end_time - start_time)
        
    mean_f1_score = np.mean(fold_f1_score)
    std_f1_score = np.std(fold_f1_score)
    mean_time = np.mean(fold_time)

    result = {}
    result['mean_f1_score'] = mean_f1_score
    result['std_f1_score'] = std_f1_score
    result['mean_time'] = mean_time

    return result

#### Main Function ####
def main(args):

    set_seed(args.seed)

    print("Load " + args.lang + " dataset...")
    positive_data = []
    negative_data = []
    with open(f'dataset/{args.lang}.jsonl', 'r') as f:
        for line in f:
            inst = json.loads(line)
            if inst['label'] == 1:
                positive_data.append(inst)
            else:
                negative_data.append(inst)
    positive_file_name = []
    negative_file_name = []
    for inst in positive_data:
        positive_file_name.append(inst['file_name'])
    for inst in negative_data:
        negative_file_name.append(inst['human_file_name'] + inst['llm_file_name'])
    positive_file_name = list(set(positive_file_name))
    negative_file_name = list(set(negative_file_name))
    print("Create K-Fold Datasets...")
    # create kfold datasets
    positive_kfold_file_names = create_randomized_kfold_datasets(positive_file_name, k=args.k, seed=args.seed)
    negative_kfold_file_names = create_randomized_kfold_datasets(negative_file_name, k=args.k, seed=args.seed)
    
    kfold_dataset_src = {}
    for i in range(args.k):
        positive_train_file_names, positive_test_file_names = positive_kfold_file_names[i]
        negative_train_file_names, negative_test_file_names = negative_kfold_file_names[i]
        train_src = []
        test_src = []
        for inst in positive_data:
            if inst['file_name'] in positive_train_file_names:
                train_src.append(inst)
            else:
                test_src.append(inst)
        for inst in negative_data:
            if inst['human_file_name'] + inst['llm_file_name'] in negative_train_file_names:
                train_src.append(inst)
            else:
                test_src.append(inst)
        item_src = {}
        item_src['train'] = train_src
        item_src['test'] = test_src
        kfold_dataset_src[i] = item_src

    kfold_dataset_feature = {}
    for i in range(args.k):
        positive_train_file_names, _ = positive_kfold_file_names[i]
        negative_train_file_names, _ = negative_kfold_file_names[i]
        train_feature = []
        test_feature = []
        for inst in positive_data:
            if  inst['file_name'] in positive_train_file_names:
                train_feature.append(inst)
            else:
                test_feature.append(inst)
        for inst in negative_data:
            if inst['human_file_name'] + inst['llm_file_name'] in negative_train_file_names:
                train_feature.append(inst)
            else:
                test_feature.append(inst)
        item_feature = {}
        item_feature['train'] = train_feature
        item_feature['test'] = test_feature
        kfold_dataset_feature[i] = item_feature

    # Run experiment
    total_results = {}
    print("Start K-Fold Cross Validation...")
    ml_models = ['MLPClassifier']
    print("ML Feature Experiment")
    for ml_model in ml_models:
        ml_feature_results = ml_feature_prediction(kfold_dataset_feature, ml_model, args)
        total_results[ml_model] = ml_feature_results

    # Save Total Results
    with open(f'{args.lang}_results.pkl', 'wb') as f:
        pickle.dump(total_results, f)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lang', type=str, default="c")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--k', type=int, default=5)

    args = parser.parse_args()

    main(args)