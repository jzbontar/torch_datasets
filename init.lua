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
   local bin_path = path .. '/cifar_' .. preprocess .. '.t7'

   if not paths.filep(bin_path) then
      -- os.execute('wget -N -P ' .. path ..' ' .. 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz')
      -- os.execute('tar xvzf ' .. path .. '/cifar-10-binary.tar.gz -C ' .. path)

      local fnames = {'data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin','data_batch_4.bin','data_batch_5.bin','test_batch.bin'}

      local X = torch.DoubleTensor(60000, 3, 32, 32)
      local y = torch.DoubleTensor(60000)

      local tmp_storage = torch.DoubleStorage(10000 * (1 + 1024 * 3))
      for i, fname in ipairs(fnames) do
         local f = torch.ByteStorage(path .. '/cifar-10-batches-bin/' .. fname)
         tmp_storage:copy(f)
         local tmp_tensor = torch.DoubleTensor(tmp_storage, 1, torch.LongStorage{10000, 1 + 1024 * 3})
         X:narrow(1, (i - 1) * 10000 + 1, 10000):copy(tmp_tensor:narrow(2, 2, 1024 * 3))
         y:narrow(1, (i - 1) * 10000 + 1, 10000):copy(tmp_tensor:narrow(2, 1, 1))
      end
      y:add(1)

      if preprocess == 1 then
         -- Torch7 demo
         normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
         for i = 1,X:size(1) do
            local rgb = X[i]
            local yuv = image.rgb2yuv(rgb)
            yuv[1] = normalization(yuv[{{1}}])
            X[i] = yuv
         end

         mean = X[{{},2,{},{}}]:mean()
         std = X[{{},2,{},{}}]:std()
         X[{{},2,{},{}}]:add(-mean):div(std)

         mean = X[{{},3,{},{}}]:mean()
         std = X[{{},3,{},{}}]:std()
         X[{{},3,{},{}}]:add(-mean):div(std)
      elseif preprocess == 2 then
         -- Alex
         mean = torch.DoubleTensor(3072)
         f = torch.DiskFile(path .. '/mean')   
         f:readDouble(mean:storage()) 
         f:close()

         X:resize(60000, 3072)
         X:add(-1, mean:repeatTensor(60000, 1))
         X:resize(60000, 3, 32, 32)
      elseif preprocess == 3 then
         -- Jure
         X2 = torch.Tensor(60000, 1, 32, 32)
         for i = 1,60000 do
            X2[{i, 1}]:copy(image.rgb2y(X[i]))
         end
         X = X2

         X:resize(60000, 1024)
         mean = X:narrow(1, 1, 50000):mean(1)
         std = X:narrow(1, 1, 50000):std(1)
         X:add(-1, mean:repeatTensor(60000, 1))
         X:cdiv(torch.repeatTensor(std, 60000, 1))
         X:resize(60000, 1, 32, 32)
      end
      torch.save(bin_path, {X, y})
   end
   return unpack(torch.load(bin_path))
end

-- torch_datasets.cifar(3)
