require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  -- ...something here...
  local sign = torch.gt(input, 0):double()
  self.output:cmul(sign:cmul(input))
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- ...something here...
  self.gradInput:cmul(torch.gt(input,0):double()):cmul(input):mul(2)
  return self.gradInput
end 




