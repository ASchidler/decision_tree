import os

for r, d, f in os.walk("trees/v"):
    if len(os.path.split(r)[-1]) <= 2:
        for cf in f:
            if cf.endswith(".info"):
                file_parts = cf.split(".")
                flags = file_parts[3]
                try:
                    int(flags[0])
                    if flags.startswith("None1"):
                        new_filename = f"{file_parts[0]}.{file_parts[1]}.{file_parts[2]}.v{flags.replace('v', '')}.{file_parts[4]}.info"
                        os.rename(os.path.join(r, cf), os.path.join(r, new_filename))
                except ValueError:
                    pass


# for r, d, f in os.walk("trees/"):
#     if len(os.path.split(r)[-1]) <= 2:
#         for cf in f:
#             if cf.endswith(".info"):
#                 file_parts = cf.split(".")
#                 flags = file_parts[3]
#                 try:
#                     int(flags[0])
#                 except ValueError:
#                     encoding = None
#                     enc = flags[-1]
#                     new_flags = ""
#                     mode = "1"
#                     if file_parts[2] == "e":
#                         mode = "0"
#                         file_parts[4] = "e"
#                     if enc == "0":
#                         encoding = 0
#                     elif enc == "a":
#                         encoding = 1
#                     elif enc == "b":
#                         encoding = 7
#                     elif enc == "y":
#                         encoding = 2
#                     elif enc == "s":
#                         encoding = 3
#                     elif enc == "c":
#                         flags += "c"
#                         encoding = 0
#
#                     new_flags += f"{encoding}{mode}" + flags[:-1]
#                     new_filename = f"{file_parts[0]}.{file_parts[1]}.{file_parts[2]}.{new_flags}.{file_parts[4]}.info"
#                     os.rename(os.path.join(r, cf), os.path.join(r, new_filename))
