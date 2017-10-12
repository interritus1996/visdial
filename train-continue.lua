require 'nn'
require 'nngraph'
require 'rnn'

------------------------------------------------------------------------
-- Input arguments and options
------------------------------------------------------------------------
local opt = require 'opts';
--print(opt)

-- seed for reproducibility
torch.manualSeed(1234);

--opt.loadPath = "checkpoints/model-10-9-2017-18:39:40-lf-att-ques-im-hist-disc/model_epoch_15.t7"
--local savedModel = torch.load(opt.loadPath)

--local modelParams = savedModel.modelParams;
--modelParams.learningRate = opt.learningRate;

--print(modelParams)
------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, {'train', 'val'});
collectgarbage();

-- set default tensor based on gpu usage
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.setDevice(opt.gpuid+1)
    cutorch.manualSeed(1234)
    torch.setdefaulttensortype('torch.CudaTensor');
else
    torch.setdefaulttensortype('torch.FloatTensor');
end

------------------------------------------------------------------------
-- Setting model parameters
------------------------------------------------------------------------
-- transfer all options to model
opt.loadPath = "checkpoints/model-10-9-2017-18:39:40-lf-att-ques-im-hist-disc/model_epoch_15.t7"
local savedModel = torch.load(opt.loadPath)

local modelParams = savedModel.modelParams;
modelParams.learningRate = opt.learningRate;

--print(opt)
print(modelParams)

-- path to save the model
local modelPath = opt.savePath

-- creating the directory to save the model
paths.mkdir(modelPath);

-- Iterations per epoch
modelParams.numIterPerEpoch = math.ceil(modelParams.numTrainThreads /
                                                modelParams.batchSize);
print(string.format('\n%d iter per epoch.', modelParams.numIterPerEpoch));

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'model'
local model = Model(modelParams);

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
print('Training..')
collectgarbage()

runningLoss = 0;
for iter = 49681, modelParams.numEpochs * modelParams.numIterPerEpoch do
    -- forward and backward propagation
    model:trainIteration(dataloader);

    -- evaluate on val and save model
    if iter % (3 * modelParams.numIterPerEpoch) == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch

        -- save model and optimization parameters
        torch.save(string.format(modelPath .. 'model_epoch_%d.t7', currentEpoch),
                                                    {modelW = model.wrapperW,
                                                    optims = model.optims,
                                                    modelParams = modelParams})
        -- validation accuracy
        model:retrieve(dataloader, 'val');
    end

    -- print after every few iterations
    if iter % 100 == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch;

        -- print current time, running average, learning rate, iteration, epoch
        print(string.format('[%s][Epoch:%.02f][Iter:%d][Loss:%.05f][lr:%f]',
                                os.date(), currentEpoch, iter, runningLoss,
                                            model.optims.learningRate))
    end
    if iter % 10 == 0 then collectgarbage(); end
end

-- Saving the final model
torch.save(modelPath .. 'model_final.t7', {modelW = model.wrapperW:float(),
                                            modelParams = modelParams});
