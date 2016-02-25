require 'torch';
require 'optim';
require 'nn';

data = torch.Tensor{
    {40, 6, 4},
    {44, 10,  4},
    {46, 12,  5},
    {48, 14,  7},
    {52, 16,  9},
    {58, 18, 12},
    {60, 22, 14},
    {68, 24, 20},
    {74, 26, 21},
    {80, 32, 24}
}

model = nn.Sequential()

ninputs = 2; noutputs = 1

model:add(nn.Linear(ninputs, noutputs))

criterion = nn.MSECriterion()


x, dl_dx = model:getParameters()

feval = function(x_new)
    if x ~= x_new then
        x:copy(n_new)
    end
    
    -- select a new training sample 
    _nindx_ = (_nindx_ or 0) + 1
    
    if _nindx_ > (#data)[1] then _nindx_ = 1 end
    
    local sample = data[_nindx_]
    local target = sample[{ {1} }]
    local inputs = sample[{ {2, 3} }]
    
    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    dl_dx:zero()
    
    --evaluate the loss function and its derivates wrt x, for the sample
    local loss_x = criterion:forward(model:forward(inputs), target)
    model:backward(inputs, criterion:backward(model.output, target))
    
    return loss_x, dl_dx 
end

sgd_params = {
    learningRate = 1e-3,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}

for i = 1, 1e3 do
    current_loss = 0
    
    -- an epoch is a full loop over our training data
    
    for i = 1, (#data)[1] do
        _, fs = optim.sgd(feval, x, sgd_params)
        
        current_loss = current_loss + fs[1]
    end
    
    current_loss = current_loss / (#data)[1]
    --print('Current loss = ' .. current_loss)
    
end 

text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}

print('id approx text')

for i = 1, (#data)[1] do
    local myPrediction = model:forward(data[i][{{2,3}}])
    print(string.format("%2d %6.2f %6.2f", i, myPrediction[1], text[i]))
end 

-- Oxford DL course assignment 2 question 2 
-- the answer is 40.10, 43.88, 49.89 for 1e4 epochs
-- for 1e5 epochs, the answer is 40.33, 44.03, 49.96
-- for 1e3 epochs, the answer is 33.07, 40.30, 46.95
-- we can see, the accuracy reach a plaute after 1e4 epochs
dataTest = torch.Tensor{
    {6, 4},
    {10, 5},
    {14, 8}
}

for i = 1, (#dataTest)[1] do
    local newPrediction = model:forward(dataTest[i])
    print(string.format('%d %6.2f', i, newPrediction[1]))
end 


-- Oxford DL course assignment 2 question 3
-- 

input = data[{{},{2, 3}}]
target = data[{{}, {1}}]

-- calculate and print the result using least squares solution 
theta = torch.inverse(input:t() * input) * input:t() * target
print('--------------- Result using Least Square Solution -----------')
for i = 1, (#data)[1] do 
    local another_prediction =  (data[i][{{2,3}}]):dot(theta:t()[1]) 
    print(string.format('%d %6.2f %6.2f', i, another_prediction, text[i]))
end
