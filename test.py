import torch
import padding

flag=True
flag=False

#x1 = torch.rand(1, 3, 50, 50)
##print(x1.squeeze(0).squeeze(0))
#print(x1)
#y = padding.padh_cpu_forward(x1, 1, flag)
##print(y.squeeze(0).squeeze(0))
#print(y)
#
#print('-'*10)

x2 = torch.ones(1, 3, 48, 72).float()
print(x2)
#print(x2.squeeze(0).squeeze(0))
print('-'*10)
y = padding.padh_cpu_backward(x2, 1, 0, flag)
#print(y.squeeze(0).squeeze(0))
print(y)


#y = padding.padh_gpu_forward(x1.cuda(), 1, flag).cpu()
##print(y.squeeze(0).squeeze(0))
#print(y)

print('-'*10)

y = padding.padh_gpu_backward(x2.cuda(), 1, 0, flag).cpu()
#print(y.squeeze(0).squeeze(0))
print(y)

