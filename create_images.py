import os
filename = 'grace_hopper.jpg'
no_of_images=10
for i in range(0,no_of_images):
	cmd = 'cp {}{} {}{}.jpg'.format('automation/tenk_images/',filename, 'automation/tenk_images/', i)
	os.system(cmd)