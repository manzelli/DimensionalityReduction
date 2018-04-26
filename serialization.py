import pickle

pkl_file = open('data.pkl', 'wb+')

serializer = pickle

x = [1,2,3,4,5,6,7,8]

serializer.dump(x,pkl_file)

pkl_file.close()

pkl_file = open('data.pkl', 'rb+')

read = pickle.load(pkl_file)

print(read)

