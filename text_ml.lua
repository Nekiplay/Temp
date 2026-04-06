local djl = require("djl")

local TextAI = {}
TextAI.__index = TextAI

-- 1. Инициализация объекта
function TextAI.new(model_id, config)
    local self = setmetatable({}, TextAI)
    self.id = model_id
    if config then
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.layers = config.layers
        self.mode = config.mode or "regression"
    end
    self.vocab = { ["<PAD>"] = 0, ["<UNK>"] = 1 }
    self.id_to_word = { [0] = "<PAD>", [1] = "<UNK>" }
    self.next_id = 2
    return self
end

-- 2. Создание модели в DJL (Обертка над djl.create_model)
function TextAI:create()
    if not self.layers or not self.input_size then
        print("[Error] Нельзя создать модель без конфигурации (layers, input_size)!")
        return false
    end
    
    print("[Create] Создание модели DJL: " .. self.id)
    return djl.create_model(self.id, {
        input_size = self.input_size,
        output_size = self.output_size,
        layers = self.layers,
        mode = self.mode
    })
end

-- 3. Внутренняя чистка текста
function TextAI:_clean(text)
    if not text then return "" end
    return text
end

-- 4. Сбор слов для словаря
function TextAI:fit(text)
    text = self:_clean(text)
    for word in text:gmatch("%S+") do
        if not self.vocab[word] then
            self.vocab[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id = self.next_id + 1
        end
    end
end

-- 5. Текст -> Числа
function TextAI:encode(text, length)
    text = self:_clean(text)
    local tokens = {}
    for word in text:gmatch("%S+") do
        table.insert(tokens, self.vocab[word] or self.vocab["<UNK>"])
        if #tokens == length then break end
    end
    while #tokens < length do table.insert(tokens, 0) end
    return tokens
end

-- 6. Числа -> Текст
function TextAI:decode(ids)
    local words = {}
    for _, id in ipairs(ids) do
        local clean_id = math.floor(id + 0.5)
        local word = self.id_to_word[clean_id] or "<UNK>"
        if word ~= "<PAD>" then table.insert(words, word) end
    end
    return table.concat(words, " ")
end

-- 7. СОХРАНЕНИЕ
function TextAI:save(path)
    print("[Save] Сохранение конфигурации: " .. path .. ".config")
    local file, err = io.open(path .. ".config", "w")
    if not file then 
        print("[Save Error] Не удалось открыть файл для записи: " .. tostring(err))
        return false 
    end

    file:write(self.input_size .. "\n")
    file:write(self.output_size .. "\n")
    file:write(self.mode .. "\n")
    file:write(table.concat(self.layers, ",") .. "\n")
    
    for word, id in pairs(self.vocab) do
        file:write(word .. "\t" .. id .. "\n")
    end
    file:close()

    print("[Save] Конфиг сохранен. Сохранение весов через DJL...")
    return djl.save_model(self.id, path)
end

-- 8. ЗАГРУЗКА
function TextAI:load(path)
    print("[Load] Чтение конфигурации: " .. path .. ".config")
    local file, err = io.open(path .. ".config", "r")
    if not file then 
        print("[Load Error] Файл не найден: " .. tostring(err))
        return false 
    end
    
    self.input_size = tonumber(file:read("*l"))
    self.output_size = tonumber(file:read("*l"))
    self.mode = file:read("*l")
    
    local layers_str = file:read("*l")
    self.layers = {}
    if layers_str then
        for n in layers_str:gmatch("%d+") do
            table.insert(self.layers, tonumber(n))
        end
    end
    
    self.vocab = {}
    self.id_to_word = {}
    for line in file:lines() do
        local word, id = line:match("^(.*)\t(%d+)$")
        if word and id then
            id = tonumber(id)
            self.vocab[word] = id
            self.id_to_word[id] = word
            if id >= self.next_id then self.next_id = id + 1 end
        end
    end
    file:close()

    local model_config = {
        input_size = self.input_size,
        output_size = self.output_size,
        layers = self.layers,
        mode = self.mode
    }

    print("[Load] Загрузка весов в модель: " .. self.id)
    return djl.load_model(self.id, path, model_config)
end

return TextAI