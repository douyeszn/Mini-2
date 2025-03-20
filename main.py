from base_iris_lab1 import load_local, build,train, score


dataset = load_local()
print(dataset)

model = build()
print(model)

hist = train(model,dataset)
print(hist)

score(model)