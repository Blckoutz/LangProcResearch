#For everything coded below, all files from pytorch demo github required
#import torch

#N, D =3, 4
#x= torch.rand((N,D), requires_grad=True)
#y= torch.rand((N,D), requires_grad=True)
#z= torch.rand((N,D), requires_grad=True)

#a=x*y
#b=a+z
#c=torch.sum(b)

#c.backward()


import string, pickle
all_letters = string.ascii_letters+".,;'"
names_by_language = pickle.load(open("names_by_language.pkl","rb"))
languages = list(names_by_language.keys())
num_names = sum(len(names) for names in names_by_language.values())

#names_by_language is a dictionary of languages, each with a list of names
print("Languages:", languages, end="\n\n")
print("First 10 English names:", names_by_language["English"][:10], end="\n\n")
print(num_names, "names in total")
