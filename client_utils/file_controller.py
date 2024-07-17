# import json
# import os
# import pandas as pd

# class FileController:
#     """
#     This class is designed to handle client features.
#     """

#     def get_file(self,file_name, type = "json"):
#         """
#         Return the last dictionary of client features from the JSON file.
#         If the file doesn't exist, create a new file with an empty dictionary and return it.
#         """
#         if type == "json":
#             file_path = os.path.join('./output', f"{file_name}.json")
#             self._check_file_availability(file_path)
#             with open(file_path, 'r') as file:
#                 # Read the file line by line in reverse order
#                 lines = file.readlines()
#                 for line in reversed(lines):
#                     try:
#                         # Try to parse JSON from the current line
#                         last_features = json.loads(line.strip())
#                         break  # Stop parsing when the last valid JSON object is found
#                     except json.JSONDecodeError:
#                         # Ignore lines that are not valid JSON
#                         pass
#             return last_features
#         else:
#             file_path = os.path.join('./output', f"{file_name}.csv")
#             df = pd.read_csv(file_path)
#             return df

#     def save_file(self, data, file_name, type="json"):
#         """
#         Take dict of features as an input and append it to the JSON file.
#         """
#         if type == "json":
#             file_path = os.path.join('./output', f"{file_name}.json")
#             self._check_file_availability(file_path)
#             with open(file_path, 'w') as file:
#                 file.write('\n')  # Ensure each dict is written on a new line
#                 json.dump(data, file)
#         else:
#             file_path = os.path.join('./output', f"{file_name}.csv")
#             data.to_csv(file_path)

#     def _check_file_availability(self,file_path):
#         if not os.path.exists(file_path):
#             os.makedirs(os.path.dirname(file_path), exist_ok=True)
#             with open(file_path, 'w') as file:
#                 json.dump({}, file)
import json
import os
import pandas as pd
import numpy as np

class FileController:
    """
    This class is designed to handle client features.
    """

    def get_file(self, file_name, file_type="json"):
        """
        Return the last dictionary of client features from the JSON file.
        If the file doesn't exist, create a new file with an empty dictionary and return it.
        """
        if file_type == "json":
            file_path = os.path.join('./output', f"{file_name}.json")
            self._check_file_availability(file_path)
            with open(file_path, 'r') as file:
                # Read the file line by line in reverse order
                lines = file.readlines()
                for line in reversed(lines):
                    try:
                        # Try to parse JSON from the current line
                        last_features = json.loads(line.strip())
                        break  # Stop parsing when the last valid JSON object is found
                    except json.JSONDecodeError:
                        # Ignore lines that are not valid JSON
                        pass
            return last_features
        elif file_type == "csv":
            file_path = os.path.join('./output', f"{file_name}.csv")
            df = pd.read_csv(file_path)
            return df
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def save_file(self, data, file_name, file_type="json"):
        """
        Take dict of features as an input and append it to the JSON file.
        """
        if file_type == "json":
            file_path = os.path.join('./output', f"{file_name}.json")
            self._check_file_availability(file_path)
            with open(file_path, 'a') as file:  # Append to the file
                file.write('\n')  # Ensure each dict is written on a new line
                json.dump(self._convert_to_serializable(data), file)
        elif file_type == "csv":
            file_path = os.path.join('./output', f"{file_name}.csv")
            if isinstance(data, dict):
                # Convert dictionary to DataFrame before saving as CSV
                data = pd.DataFrame([data])
            data.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _convert_to_serializable(self, obj):
        """
        Recursively convert non-serializable types to serializable types.
        """
        if isinstance(obj, dict):
            return {self._convert_to_serializable(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(i) for i in obj)
        elif isinstance(obj, set):
            return {self._convert_to_serializable(i) for i in obj}
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        else:
            return obj


    def _check_file_availability(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump({}, file)
