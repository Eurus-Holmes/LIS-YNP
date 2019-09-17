import learn2learn as l2l

mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)

mnist = l2l.data.MetaDataset(mnist)
task_generator = l2l.data.TaskGenerator(mnist,
                                        ways=3,
                                        classes=[0, 1, 4, 6, 8, 9],
                                        tasks=10)
model = Net()
maml = l2l.algorithms.MAML(model, lr=1e-3, first_order=False)
opt = optim.Adam(maml.parameters(), lr=4e-3)

for iteration in range(num_iterations):
    learner = maml.clone()  # Creates a clone of model
    adaptation_task = task_generator.sample(shots=1)

    # Fast adapt
    for step in range(adaptation_steps):
        error = compute_loss(adaptation_task)
        learner.adapt(error)

    # Compute evaluation loss
    evaluation_task = task_generator.sample(shots=1,
                                            task=adaptation_task.sampled_task)
    evaluation_error = compute_loss(evaluation_task)

    # Meta-update the model parameters
    opt.zero_grad()
    evaluation_error.backward()
    opt.step()
