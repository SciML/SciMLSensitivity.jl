using ForwardDiff, SciMLSensitivity, Test

intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
tag = ForwardDiff.Tag{typeof(intervals), Float64}

@test SciMLSensitivity.Gaussfindcursor(intervals, 3.0) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(3.0, 1.0),
) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(3.0, -1.0),
) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(2.5, 1.0),
) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(0.0, -1.0),
) == 1
