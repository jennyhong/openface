-- Model: fitnets1.def.lua
-- Description: Student network for FitNets branch
-- Input size: 3x96x96
-- Components: Mostly `nn`
-- Devices: CPU and CUDA
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

imgDim = 96

function createModel()
   local net = nn.Sequential()

   -- Layer 1: Convolution with filter size 7, step size 2, padding 3
   -- Output size: 48 x 48 x 32 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(3, 32, 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(32))
   net:add(nn.ReLU())

   -- Layer 4: Convolution with filter size 7, step size 2, padding 3
   -- Output size: 24 x 24 x 64 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(32, 64, 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())

   -- Layer 7: Convolution with filter size 5, step size 1, padding 2
   -- Output size: 24 x 24 x 192 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(64, 192, 5, 5, 1, 1, 2, 2))
   net:add(nn.SpatialBatchNormalization(192))
   net:add(nn.ReLU())

   -- Layer 10: Convolution with filter size 5, step size 1, padding 2
   -- Output size: 24 x 24 x 256 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(192, 256, 5, 5, 1, 1, 2, 2))
   net:add(nn.SpatialBatchNormalization(256))
   net:add(nn.ReLU())

   -- Layer 13: Convolution with filter size 5, step size 2, padding 2
   -- Output size: 12 x 12 x 256 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(256, 256, 5, 5, 2, 2, 2, 2))
   net:add(nn.SpatialBatchNormalization(256))
   net:add(nn.ReLU())

   -- Layer 16: Convolution with filter size 3, step size 1, padding 1
   -- Output size: 12 x 12 x 320 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(256, 320, 3, 3, 1, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(320))
   net:add(nn.ReLU())

   -- Layer 19: Convolution with filter size 3, step size 2, padding 1
   -- Output size: 6 x 6 x 640 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(320, 640, 3, 3, 2, 2, 1, 1))
   net:add(nn.SpatialBatchNormalization(640))
   net:add(nn.ReLU())

   -- Layer 22: Convolution with filter size 3, step size 2, padding 1
   -- Output size: 3 x 3 x 640 (width, height, depth)
   net:add(nn.SpatialConvolutionMM(640, 640, 3, 3, 2, 2, 1, 1))
   net:add(nn.SpatialBatchNormalization(640))
   net:add(nn.ReLU())

   net:add(nn.SpatialAveragePooling(3, 3))

   -- TODO: THIS IS UNFINISHED! FINISH WITH FC LAYER
   -- Validate shape with:
   -- net:add(nn.Reshape(736))

   -- what is a view
   -- net:add(nn.View(736))
   -- net:add(nn.Linear(736, opt.embSize))
   -- net:add(nn.Normalize(2))

   return net
end
