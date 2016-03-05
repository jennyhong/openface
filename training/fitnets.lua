#!/usr/bin/env th

require 'torch'
require 'torchx' --for concetration the table of tensors
require 'optim'

require 'paths'

require 'xlua'
require 'csvigo'

require 'nn'
require 'dpnn'

paths.dofile("FitNetsOptim.lua")
-- paths.dofile("train.lua")

local sanitize = paths.dofile('sanitize.lua')

local optimMethod = optim.adadelta
local optimState = {} -- Use for other algorithms like SGD
local optimator = FitNetsOptim(model, optimState)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local total_loss




teacher_model = torch.load('models/openface/nn4.small2.v1.t7')
teacher_layer = 1

function fitnetsTrainBatch(inputsThread, numPerClassThread)
  if batchNumber >= opt.epochSize then
    return
  end

  if opt.cuda then
    cutorch.synchronize()
  end
  timer:reset()
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(numPerClassThread, numPerClass)

  local inputs
  if opt.cuda then
     inputs = inputsCPU:cuda()
  else
     inputs = inputsCPU
  end

  local numImages = inputs:size(1)

  local student_embeddings = model:forward(inputs):float()
  local teacher_final_embeddings = teacher_model:forward(inputs):float()
  local teacher_middle_embeddings = teacher_model.modules[teacher_layer].output
  local output = teacher_middle_embeddings

  local criterion = nn.MSECriterion()

  local err, _ = optimator:optimize(optimMethod, inputs, output, criterion)

  -- DataParallelTable's syncParameters
  model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
  if opt.cuda then
     cutorch.synchronize()
  end
  batchNumber = batchNumber + 1
  print(('Epoch: [%d][%d/%d]\tTime %.3f\ttotalErr %.2e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err))
  timer:reset()
  total_loss = total_loss + err
end