require 'torch'
require 'paths'

base_path = '/home/jure/datasets'

torch_datasets = {}
function torch_datasets.mnist()
   local path = base_path .. '/mnist'
   local bin_path = path .. '/mnist.th'

   if not paths.filep(bin_path) then
      os.execute('wget -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
      os.execute('wget -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
      os.execute('wget -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
      os.execute('wget -P ' .. path .. ' ' .. 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
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
