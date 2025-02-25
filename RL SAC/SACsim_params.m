function SACsim_params(model)

% Validar las entradas
if ~ismember(model, {'M0', 'M3'})
    error('Model must be: ''M0'', ''M3''.');
end

switch model
    case 'M0'
        ScalerInMin=[-5.9009795 1287.4781 -54.729496 -3.403699 -4.377029 0.];
        ScalerInMax=[1.5060300e+02 2.8380360e+03 8.5859970e+00 5.1123200e+01 4.4160643e+00 1.2493309e+00];
        
        ScalerOutMin=[3.5043920e-01 1.5963114e-10 3.6094947e-13 -7.3952870e-02];
        ScalerOutMax=[8.17552600e-01 3.79204100e-06 1.99143300e-09 1.02194086e-01];
        ScalerOut_range=[4.67113400e-01 3.79188137e-06 1.99107205e-09 1.76146956e-01];
        
        assignin('base', 'ScalerInMin', ScalerInMin);
        assignin('base', 'ScalerInMax', ScalerInMax);
        assignin('base', 'ScalerOutMin', ScalerOutMin);
        assignin('base', 'ScalerOutMax', ScalerOutMax);
        assignin('base', 'ScalerOut_range', ScalerOut_range);

    case 'M3'
        ScalerMin=[-4.6178937e-01, 1.2874781e+03, -5.4729496e+01, -3.7608946e-02, -4.3601995e+00, 0.0000000e+00, 3.5390502e-01, 1.0100000e-09, 2.1700000e-12, -7.3952870e-02, 3.5441822e-01, 1.6000000e-10, 3.6100000e-13, -7.3952870e-02];
        ScalerMax=[1.5060300e+02, 2.8380360e+03, 4.8536170e+00, 5.1123200e+01, 4.3944810e+00, 1.2493309e+00, 8.1755260e-01, 3.7900000e-06, 6.3800000e-10, 9.3600790e-02, 8.1755260e-01, 3.7900000e-06, 6.3800000e-10, 9.3600790e-02];
        
        SOCscaling_min=[ScalerMin(1:6),ScalerMin(11),ScalerMin(14)];
        SOCscaling_max=[ScalerMax(1:6),ScalerMax(11),ScalerMax(14)];
        
        NOscaling_min=[ScalerMin(1:6),ScalerMin(12),ScalerMin(14)];
        NOscaling_max=[ScalerMax(1:6),ScalerMax(12),ScalerMax(14)];
        
        NO2scaling_min=[ScalerMin(1:6),ScalerMin(13),ScalerMin(14)];
        NO2scaling_max=[ScalerMax(1:6),ScalerMax(13),ScalerMax(14)];
        
        NOxscaling_min=[ScalerMin(1:6),ScalerMin(12),ScalerMin(13),ScalerMin(14)];
        NOxscaling_max=[ScalerMax(1:6),ScalerMax(12),ScalerMax(13),ScalerMax(14)];

        assignin('base', 'SOCscaling_min', SOCscaling_min);
        assignin('base', 'SOCscaling_max', SOCscaling_max);
        assignin('base', 'NOscaling_min', NOscaling_min);
        assignin('base', 'NOscaling_max', NOscaling_max);
        assignin('base', 'NO2scaling_min', NO2scaling_min);
        assignin('base', 'NO2scaling_max', NO2scaling_max);
        assignin('base', 'NOxscaling_min', NOxscaling_min);
        assignin('base', 'NOxscaling_max', NOxscaling_max);
end

