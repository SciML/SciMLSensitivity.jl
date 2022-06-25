using SciMLSensitivity, Test

@test SciMLSensitivity.hasbranching(1, 2) do x, y
   (x < 0 ? -x : x) + exp(y)
end

@test !SciMLSensitivity.hasbranching(1, 2) do x, y
   ifelse(x < 0, -x, x) + exp(y)
end
