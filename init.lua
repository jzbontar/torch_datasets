require 'torch'
require 'paths'
require 'nn'
require 'image'

base_path = '/home/jure/datasets'

torch_datasets = {}
function torch_datasets.mnist()
   local path = base_path .. '/mnist'
   local bin_path = path .. '/mnist.t7'

   if not paths.filep(bin_path) then
      os.execute('wget -N -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
      os.execute('wget -N -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
      os.execute('wget -N -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
      os.execute('wget -N -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
      os.execute('gunzip ' .. path .. '/train-images-idx3-ubyte.gz')
      os.execute('gunzip ' .. path .. '/train-labels-idx1-ubyte.gz')
      os.execute('gunzip ' .. path .. '/t10k-images-idx3-ubyte.gz')
      os.execute('gunzip ' .. path .. '/t10k-labels-idx1-ubyte.gz')

      f_tr = torch.DiskFile(path .. '/train-labels-idx1-ubyte')
      f_te = torch.DiskFile(path .. '/t10k-labels-idx1-ubyte')

      y_tr = torch.ByteTensor(60000)
      y_te = torch.ByteTensor(10000)

      f_tr:readByte(8)
      f_tr:readByte(y_tr:storage())

      f_te:readByte(8)
      f_te:readByte(y_te:storage())

      y_tr:add(1)
      y_te:add(1)
      y_tr = y_tr:long()
      y_te = y_te:long()

      f_tr = torch.DiskFile(path .. '/train-images-idx3-ubyte')
      f_te = torch.DiskFile(path .. '/t10k-images-idx3-ubyte')

      X_tr = torch.ByteTensor(60000, 784)
      X_te = torch.ByteTensor(10000, 784)

      f_tr:readByte(16)
      f_tr:readByte(X_tr:storage())

      f_te:readByte(16)
      f_te:readByte(X_te:storage())

      X_tr = X_tr:float():div(255)
      X_te = X_te:float():div(255)

      torch.save(bin_path, {X_tr, y_tr, X_te, y_te})
   end
   return unpack(torch.load(bin_path))
end

function torch_datasets.cifar(preprocess)
   preprocess = preprocess or 1

   local path = base_path .. '/cifar'
   local bin_path = path .. '/cifar.t7'

   if not paths.filep(bin_path) then
      -- os.execute('wget -N -P ' .. path ..' ' .. 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz')
      -- os.execute('tar xvzf ' .. path .. '/cifar-10-binary.tar.gz -C ' .. path)

      local fnames = {'data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin','data_batch_4.bin','data_batch_5.bin','test_batch.bin'}

      local X = torch.FloatTensor(60000, 3, 32, 32)
      local y = torch.FloatTensor(60000)

      local tmp_storage = torch.FloatStorage(10000 * (1 + 1024 * 3))
      for i, fname in ipairs(fnames) do
         local f = torch.ByteStorage(path .. '/cifar-10-batches-bin/' .. fname)
         tmp_storage:copy(f)
         local tmp_tensor = torch.FloatTensor(tmp_storage, 1, torch.LongStorage{10000, 1 + 1024 * 3})
         X:narrow(1, (i - 1) * 10000 + 1, 10000):copy(tmp_tensor:narrow(2, 2, 1024 * 3))
         y:narrow(1, (i - 1) * 10000 + 1, 10000):copy(tmp_tensor:narrow(2, 1, 1))
      end
      y:add(1)

      if preprocess == 1 then
         nn = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
         for i = 1,X:size(1) do
            print(i)
            break
         end
      end
   end
end

torch_datasets.cifar()
