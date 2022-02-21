function [status, Out] = GetDevice(IP, Port, User, Passwd)
    Out = [];
    [status, gpu] = dos(['/usr/local/bin/sshpass -p ',Passwd,' ssh -p ',num2str(Port),' ',User,'@',IP,' nvidia-smi --list-gpus']);
    if(status==0)
        Out = struct('DeviceID',-1,'Device',{'CPU'});
        gpu = splitlines(gpu);
        for i=0:length(gpu) - 2
            Out.DeviceID = cat(1, Out.DeviceID, i);
            Out.Device = cat(1, Out.Device, {['GPU', num2str(i)]});
        end
    end
end