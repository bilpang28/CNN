# AI No. 1
# Pemrosesan Gambar dengan Conditional Neural Network

Program untuk memproses sebuah dataset gambar dari CIFAR10 sebagai dataset training dan dataset testing. Program ini menggunakan Operating System Windows.
untuk pengambilan dataset sendiri dilakukan secara online.

# Instalasi Environment 

1. Import Torch menggunakan pip
   
         pip install torch
   
2. Import Torchvision menggunakan pip
   
        pip install torchvision
   
3. Import matplotlib.pyplot menggunakan pip

        pip install matplotlib

4. Import Numpy menggunakan pip

        pip install numpy

5. Import SSL menggunakan pip

        pip install ssl

# Menghubungkan ke Dataset

  Setelah meng - Import SSL ketikkan kode berikut :
  
        ssl._create_default_https_context = ssl._create_unverified_context 

  kode ini untuk menghubungkan ke HTTPS yang diinginkan oleh client. kemudian akan diverifikasi oleh server untuk keamanan koneksi.
  
# Mengatur Gambar

Kemudian atur ukuran default gambar dengan panjang 14 inchi dan lebar 6 inchi.

      plt.rcParams['figure.figsize'] = 14, 6

# Mengatur Dataset 

Membuat komposisi untuk perubahan dataset untuk dapat meng - konversi data gambar ke PyTorch Tensor.

      normalize_transform = torchvision.transforms.Compose([ 
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean = (0.5, 0.5, 0.5),
      std = (0.5, 0.5, 0.5))])
      
# Mengunduh Dataset 

Dataset yang diunduh akan digunakan untuk melakukan Training dan Testing Program. Berikut kode untuk Me - Training :

      train_dataset = torchvision.datasets.CIFAR10( 
      	root="./CIFAR10/train", train=True, 
      	transform=normalize_transform, 
      	download=True) 

Berikut kode untuk  men - Testing  :

      test_dataset = torchvision.datasets.CIFAR10( 
      	root="./CIFAR10/test", train=False, 
      	transform=normalize_transform, 
      	download=True) 
       
# Mengatur Jumlah Batch 

Jumlah batch akan dibatasi sampai 128 untuk setiap data pada folder Training maupun Testing
      
      batch_size = 128
      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size) 
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size) 

# Visualisasi Gambar

Memvisualisasikan 25 Gambar ke dalam 5 x 5 Grid.

      dataiter = iter(train_loader) 
      images, labels = next(dataiter) 
      plt.imshow(np.transpose(torchvision.utils.make_grid( 
      images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0))) 
      plt.axis('off')

# Menyimpan data ke Label 

      classes = [] 
      for batch_idx, data in enumerate(train_loader, 0): 
          x, y = data  
          classes.extend(y.tolist()) 

# Menghitung Kelas yang Unik 

kelas yang unik akan dihitung dan dilakukan plotting

      unique, counts = np.unique(classes, return_counts=True) 
      names = list(test_dataset.class_to_idx.keys()) 
      plt.bar(names, counts) 
      plt.xlabel("Target Classes") 
      plt.ylabel("Number of training instances")

# Pembuatan Kelas CNN

Menghasilkan input dan output yang berbeda di setiap perulangan
    
      class CNN(torch.nn.Module): 
   	def __init__(self): 
   		super().__init__() 
   		self.model = torch.nn.Sequential( 
   			#Input = 3 x 32 x 32, Output = 32 x 32 x 32 
   			torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1), 
   			torch.nn.ReLU(), 
   			#Input = 32 x 32 x 32, Output = 32 x 16 x 16 
   			torch.nn.MaxPool2d(kernel_size=2), 
   
   			#Input = 32 x 16 x 16, Output = 64 x 16 x 16 
   			torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1), 
   			torch.nn.ReLU(), 
   			#Input = 64 x 16 x 16, Output = 64 x 8 x 8 
   			torch.nn.MaxPool2d(kernel_size=2), 
   			
   			#Input = 64 x 8 x 8, Output = 64 x 8 x 8 
   			torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1), 
   			torch.nn.ReLU(), 
   			#Input = 64 x 8 x 8, Output = 64 x 4 x 4 
   			torch.nn.MaxPool2d(kernel_size=2), 
   
   			torch.nn.Flatten(), 
   			torch.nn.Linear(64*4*4, 512), 
   			torch.nn.ReLU(), 
   			torch.nn.Linear(512, 10) 
   		) 
   
   	def forward(self, x): 
   		return self.model(x) 
     
# Pemilihan untuk Training data 

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      model = CNN().to(device) 
      
# Mendefinisikan model dari parameter

      num_epochs = 50
      learning_rate = 0.001
      weight_decay = 0.01
      criterion = torch.nn.CrossEntropyLoss() 
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

# Proses data di Training 

      train_loss_list = [] 
      for epoch in range(num_epochs): 
      	print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ') 
      	train_loss = 0
	
# memasukkan model Training ke dalam Batch 

	model.train() 
	for i, (images, labels) in enumerate(train_loader): 
		
# Melakukan Ekstraksasi data

		images = images.to(device) 
		labels = labels.to(device) 

# Menghitung Output 

		outputs = model(images) 
		loss = criterion(outputs, labels) 

		#Updating weights according to calculated loss 
		optimizer.zero_grad() 
		loss.backward() 
		optimizer.step() 
		train_loss += loss.item() 
	
# Mencetak setiap Epoch 

	train_loss_list.append(train_loss/len(train_loader)) 
	print(f"Training loss = {train_loss_list[-1]}") 
	
# Menampilkan output

      plt.plot(range(1,num_epochs+1), train_loss_list) 
      plt.xlabel("Number of epochs") 
      plt.ylabel("Training loss") 
      test_acc=0
      model.eval() 

      with torch.no_grad(): 
      	#Iterating over the training dataset in batches 
      	for i, (images, labels) in enumerate(test_loader): 
		
   		images = images.to(device) 
   		y_true = labels.to(device) 
		
# Menghitung Output Untuk Batch
		outputs = model(images) 
		
		#Calculated prediction labels from models 
		_, y_pred = torch.max(outputs.data, 1) 
		
# Menggabungkan label True dengan Prediksi

		test_acc += (y_pred == y_true).sum().item() 
	
	print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")
  
# Mengolah "Num_Image" untuk Prediksi
      
      num_images = 5
         y_true_name = [names[y_true[idx]] for idx in range(num_images)] 
         y_pred_name = [names[y_pred[idx]] for idx in range(num_images)] 

# Memberikan Judul 
   
      title = f"Actual labels: {y_true_name}, Predicted labels: {y_pred_name}"

# Finalisasi
         
         plt.imshow(np.transpose(torchvision.utils.make_grid(images[:num_images].cpu(), normalize=True, padding=1).numpy(), (1, 2, 0))) 
         plt.title(title) 
         plt.axis("off")
