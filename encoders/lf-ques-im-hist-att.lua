local encoderNet = {}

function encoderNet.model(params)

    local dropout = 0.5

    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) -- question
    table.insert(inputs, nn.Identity()()) -- img feats Bx14x14x512
    table.insert(inputs, nn.Identity()()) -- history

    local ques = inputs[1]
    local img_feats = inputs[2]
    local hist = inputs[3]

    -- word embed layer
    wordEmbed = nn.LookupTableMaskZero(params.vocabSize, params.embedSize);

    -- make clones for embed layer
    local qEmbed = nn.Dropout(dropout)(wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias')(ques));
    local hEmbed = nn.Dropout(dropout)(wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias')(hist));

    local lst1 = nn.SeqLSTM(params.embedSize, params.rnnHiddenSize)
    lst1:maskZero()

    local lst2 = nn.SeqLSTM(params.rnnHiddenSize, params.rnnHiddenSize)
    lst2:maskZero()

    local h1 = lst1(hEmbed)
    local h2 = lst2(h1)
    local h3 = nn.Select(1, -1)(h2)

    local lst3 = nn.SeqLSTM(params.embedSize, params.rnnHiddenSize)
    lst3:maskZero()

    local lst4 = nn.SeqLSTM(params.rnnHiddenSize, params.rnnHiddenSize)
    lst4:maskZero()

    local q1 = lst3(qEmbed)
    local q2 = lst4(q1)
    local q3 = nn.Select(1, -1)(q2)

    -- view q3 as (B*10)x512x1
    local q3_view = nn.View(-1, 512, 1):setNumInputDims(2)(q3)
    -- view img_feats as (B*10)x196x512 
    local img_feats_view = nn.Transpose({1, 2})(nn.View(-1, params.batchSize*10, 512)(nn.Transpose({1, 2}, {2, 3})(img_feats)))

    local qi_mm = nn.MM(){img_feats_view, q3_view}
    -- probabilites for 14x14 image patches 
    local probs = nn.SoftMax()(nn.View(-1, 196)(qi_mm))
    -- view Probabilites as (B*10)x196x1 
    local probs_view = nn.View(-1, 196, 1)(probs)

    --weighted sum of probablites and image features
    local att_img_feats = nn.MM(true, false){probs_view, img_feats_view}
    -- att_img_feats are the image features with attention applied over 14x14 regions of image
    local att_img_feats_view = nn.View(-1, 512)(att_img_feats)

    -- concatenating img features, ques features and hist features
    local qih = nn.JoinTable(1, 1)({att_img_feats_view, q3, h3})
    local final_layer = nn.Tanh()(nn.Linear(512*3, 512)(nn.Dropout(dropout)(qih)))

    table.insert(outputs, final_layer)

    local enc = nn.gModule(inputs, outputs)
    enc.wordEmbed = wordEmbed

    return enc;
end

return encoderNet
