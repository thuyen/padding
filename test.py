import torch
import padding

#x1 = torch.rand(1, 1, 4, 4)
#print(x1.squeeze(0).squeeze(0))
#y = padding.padh_cpu_forward(x1, 1)
#print(y.squeeze(0).squeeze(0))

print('-'*10)

x2 = torch.ones(1, 1, 6, 4).float()
print(x2.squeeze(0).squeeze(0))
print('-'*10)
y = padding.padh_cpu_backward(x2, 1)
print(y.squeeze(0).squeeze(0))


#y = padding.padh_gpu_forward(x1.cuda(), 1).cpu()
#print(y.squeeze(0).squeeze(0))

print('-'*10)

y = padding.padh_gpu_backward(x2.cuda(), 1).cpu()
print(y.squeeze(0).squeeze(0))

