import matplotlib.pyplot as plt

data1 = list(map(float, open("data/CC3M/train/train.subset.80.clipscore.csv").readlines()))
data2 = list(map(float, open("data/CC12M/real-aligned/train.clipscore.csv").readlines()))
data3 = list(map(float, open("data/CC12M/fake-aligned/train.clipscore.csv").readlines()))

plt.figure(figsize = (8,6))
plt.hist(data1, bins = 100, alpha = 0.5, label = "CC3M")
plt.hist(data2, bins = 100, alpha = 0.5, label = "CC12M-Real")
plt.hist(data3, bins = 100, alpha = 0.5, label = "CC12M-Fake")
plt.xlabel("CLIP Score", size = 14)
plt.ylabel("Count", size = 14)
plt.title("Pretrained OpenAI's CLIP score for Image-Caption Data")
plt.legend(loc = "upper right")
plt.savefig("clipscore.png")