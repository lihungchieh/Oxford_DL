require 'nngraph'
require 'nn'

--z = x_1 + x_2 .* (linear(x3))

input1 = nn.Identity()()
input2 = nn.Identity()()
input3 = nn.Identity()()

result = nn.CAddTable()({input1, nn.CMulTable()({input2, nn.Linear(4,10)(input3)})})

m = nn.gModule({input1, input2, input3}, {result})

a = torch.Tensor{1,2,3,4,5,6,7,8,9,10}
b = torch.Tensor{2,2,2,2,2,2,2,2,2,2}
c = torch.Tensor{1,2,3,4}

target = torch.Tensor{13, 12, 33, 42, 51, 16, 17, 18, 19, 20}

result = m:updateOutput({a, b, c})
print(result)
--graph.dot(m.fg, 'Try NNGraph')

local parameters, gradParameters = m:getParameters()

criterion = nn.MSECriterion()

require 'optim'
optimState = {
    learningRate = 1e-3,
    weightDecay = 1e-4,
    momentum = 0,
    learningRateDecay = 1e-7
  }

local feval = function(x)
    if x ~= parameters then
        parameters:copy(x)
    end
    
    local batch_inputs = {a, b, c}
    local batch_targets = target
    gradParameters:zero()
    
    local batch_outputs = m:forward(batch_inputs)
    local batch_loss = criterion:forward(batch_outputs, batch_targets)
    
    local dloss_doutput = criterion:backward(batch_outputs, batch_targets)
    
    m:backward(batch_inputs, dloss_doutput)
    
    return batch_loss, gradParameters
end 

local losses = {} 
local iterations = 1000

for i = 1, iterations do 
    local _, minibatch_loss = optim.sgd(feval, parameters, optimState)
    
    if i % 10 == 0 then
        print(string.format('minibatches processed %6s, loss = %6.6f', i, minibatch_loss[1]))
    end
end

myoutput = m:forward({a, b, c})

for i = 1, 10 do 
    print(string.format("Predictions %6.6f , True Value %6d", myoutput[i], target[i]))
end 
    