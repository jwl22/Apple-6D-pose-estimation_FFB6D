f = open("apple_train_data_list.txt", "w+")
# for i in range(20):
# for i in range(30950):
#     f.write(f"data/30949/{str(i).zfill(6)} \n")
# for i in range(30651):
#     f.write(f"data/30650/{str(i).zfill(6)} \n")
# for i in range(54774):
#     f.write(f"data/{str(i).zfill(6)} \n")
# for i in range(50000):
#     f.write(f"data_syn/{str(i).zfill(6)} \n")

for i in range(10000):
    f.write(f"test/{str(i).zfill(6)} \n")
