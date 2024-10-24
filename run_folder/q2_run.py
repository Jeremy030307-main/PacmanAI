import os, subprocess
import re

small_maze = ['1_1', '1_2','1_3','1_4','2_1','2_2','2_3','2_4','3_1','3_2','3_3','3_4']
discount_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def command(maze_name, maze_spec):
    return[
        "python", "pacman.py", 
        "-l", f"layouts/QL_{maze_name}{maze_spec}.lay", 
        "-p", "Q2Agent", 
        "-a", f"epsilon={0.2},alpha={0.5},gamma={1.0}", 
        "-x", f"{100}",
        "-n", f"{101}",
        "-g", "StationaryGhost"
    ]

result = subprocess.run(command("tinyMaze", "1_1"))
print("Output", result.stdout)
print("Error", result.stderr)


# Define a function to run the command with retry logic
# def run_command_with_timeout(command, timeout=1300):
#     try:
#         result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
#         return result
#     except subprocess.TimeoutExpired:
#         print("Command timed out!")
#         return None  # Return None or handle timeout as needed

# result_dic = {discount: 0 for discount in discount_values}

# for spec in small_maze: 
#     print(spec, "--------------------------------")
#     for discount in discount_values:
#         print(discount)
#         result = run_command_with_timeout(command('bigMaze', spec, discount, 100))

#         # If the result is None, it means there was a timeout
#         if result is None:
#             print("Skipping due to timeout.")
#             continue  # Skip this discount value 

#         average_score_match = re.search(r'Average Score:\s*([0-9\.\-]+)', result.stdout)
#         if average_score_match:
#             average_score = average_score_match.group(1)
#             if float(average_score) > 0:
#                 result_dic[discount] += float(average_score) 

# import json

# # Writing the dictionary to a JSON file
# with open("output.json", "w") as file:
#     json.dump(result_dic, file)


