function [ObjectClassMasks, ObjectInstanceMasks, PartsClassMasks, PartsInstanceMasks, objects, parts] = loadAde20K(file)
%
% [ObjectClassMasks, ObjectInstanceMasks, PartsClassMasks, PartsInstanceMasks, objects, parts] = loadAde20K(filename);
% 
% The ouput is:
%   ObjectClassMasks    [n * m]
%   ObjectInstanceMasks [n * m]
%   PartsClassMasks     [n * m * Nlevels]
%   PartsInstanceMasks  [n * m * Nlevels]
%   objects
%   parts
%
% The first four output are segmentation masks. 
%
% The last two outputs provide complementary information:
% objects = 
%        instancendx: index inside 'ObjectInstanceMasks'
%              class: object name
%             iscrop: indicates if the objects is whole (iscrop=0) or partially visible (iscrop=1)
%     listattributes: comma separated list of attributes such as 'sitting', ...
%
% parts = 
%        instancendx: index inside 'PartsInstanceMasks(:,:,level)'
%              level: level in the part hierarchy. Level = 1 means that is
%                     a direct object part. Level = 2 means that is part of a
%                     part.
%              class: part name
%             iscrop: whole (iscrop=0) or partially visible (iscrop=1)
%     listattributes: ]attributes
% 
%
% 


fileseg = strrep(file,'.jpg', '_seg.png');

% Read object masks
seg = imread(fileseg);

R = seg(:,:,1);
G = seg(:,:,2);
B = seg(:,:,3);

ObjectClassMasks = (uint16(R)/10)*256+uint16(G);
[~,~,Minstances_hat] = unique(B(:));
ObjectInstanceMasks = reshape(Minstances_hat-1,size(B));

if nargout>2 || nargout==0
    % Read part masks
    level = 0;
    PartsClassMasks = [];
    PartsInstanceMasks = [];
    while 1
        level = level+1;
        file_parts = strrep(file,'.jpg', sprintf('_parts_%d.png', level));
        if exist(file_parts, 'file')
            partsseg = imread(file_parts);
            R = partsseg(:,:,1);
            G = partsseg(:,:,2);
            B = partsseg(:,:,3);
            PartsClassMasks(:,:,level) = (uint16(R)/10)*256+uint16(G);
            [~,~,Minstances_hat] = unique(B(:));
            PartsInstanceMasks(:,:,level) = reshape(Minstances_hat-1,size(B));
        else
            break
        end
    end
end

if nargout>4
    % Read attributes
    % format: '%03d; %s;  %d; %s; "%s"\n', instance, name, whole(j)==0, crop, atr
    % format: Instance, part level (0 for objects), crop, class name, corrected_raw_name, list-of-attributes
    fid = fopen(strrep(file, '.jpg', '_atr.txt'), 'r');
    if fid>-1
        C = textscan(fid, '%s', 'Delimiter', '#');
        fclose(fid);
        C = C{1};
        C = reshape(C, [6 length(C)/6])';
        
        instance = str2num(cell2mat(C(:,1)));
        names = strtrim(C(:,4)); % this is the wordnet name
        corrected_raw_name = strtrim(C(:,5));
        partlevel = str2num(cell2mat(C(:,2)));
        ispart = partlevel>0;
        iscrop = str2num(cell2mat(C(:,3)));
        listattributes = strtrim(strrep(C(:,6), '"', ''));
        
        objects.instancendx = instance(ispart==0);
        objects.class = names(ispart==0);
        objects.corrected_raw_name = corrected_raw_name(ispart==0);
        objects.iscrop = iscrop(ispart==0);
        objects.listattributes = listattributes(ispart==0);
        
        parts.instancendx = instance(ispart==1);
        parts.level = partlevel(ispart==1);
        parts.class = names(ispart==1);
        parts.corrected_raw_name = corrected_raw_name(ispart==1);
        parts.iscrop = iscrop(ispart==1);
        parts.listattributes = listattributes(ispart==1);
    else
        objects = [];
        parts = [];
    end
end
    
if nargout == 0
    figure
    subplot(221)
    imshow(ObjectClassMasks, []);
    title('Object class')
    subplot(222)
    imshow(ObjectInstanceMasks, []);
    title('Instances')
    subplot(223)
    imshow(sum(PartsClassMasks,3), []);
    title('Part class')
    subplot(224)
    imshow(sum(PartsInstanceMasks,3), []);
    title('Part Instance')
    colormap(cat(1, [0 0 0], hsv(255)))
end
