local djl = require("djl")
local TextAI = require("text_ml")
local threads = require("threads")
djl.set_device("gpu", 0)
-- 1. Конфигурация модели
-- Мы будем использовать 3 слова на вход и 3 слова на выход
local IN_LEN = 12
local OUT_LEN = 12
local model_id = "my_bot"

local config = {
    input_size = IN_LEN,
    output_size = OUT_LEN,
    layers =  {1024, 1024, 512, 512}, -- Два скрытых слоя
    mode = "regression"
}

-- Создаем объект нашей библиотеки
local ai = TextAI.new(model_id, config)

-- 2. Подготовка данных для обучения (Вопрос -> Ответ)
local raw_data = {
    {q = "hello", a = "Привет"},
    {q = "Hello", a = "Привет"},
	{q = "Hallo", a = "Привет"},
	{q = "Hullo", a = "Привет"},
	{q = "Howdy", a = "Привет"},
    {q = "How are you", a = "Как дела"},
    {q = "What are you doing", a = "Что делаешь"},
	{q = "What do you do?", a = "Чем занимаешся?"},
    {q = "What can you do", a = "Что ты умеешь"},
	{q = "Hi all", a = "Всем привет"},
	{q = "Hi", a = "Привет"},
	{q = "Hi, how are you", a = "Привет, как дела"},
	{q = "Hello Neki_play", a = "Привет Neki_play"},
	{q = "Hello Maxim", a = "Привет Максим"},
	{q = "PIG", a = "СВИНЬЯ"},
	{q = "Pig", a = "Свинья"},
	{q = "Sorry", a = "Прости"},
	{q = "sorry", a = "Прости"},
	{q = "Thats what i do", a = "Вот чем я занимаюсь"},
	{q = "was rekt by", a = "был разгромлен"},
	{q = "Rixxware rekt Rixxware", a = "Rixxware разгромлен Rixxware"},
	{q = "there", a = "Там"},
	{q = "seb496 sucked ThePowerishere1 dry", a = "seb496 высосал из ThePowerishere1 все соки"},
	{q = "nilla what", a = "nilla что"},
	{q = "what", a = "Что"},
	{q = "What", a = "Что"},
	{q = "dang", a = "Черт возьми"},
	{q = "Dang", a = "Черт возьми"},
	{q = "hello", a = "привет"},
    {q = "good morning", a = "доброе утро"},
    {q = "good night", a = "спокойной ночи"},
    {q = "how are you", a = "как твои дела"},
   
   -- Приветствия и Вежливость
    {q = "How are you", a = "Как твои дела"},
    {q = "I am fine", a = "Я в порядке"},
    {q = "Hello my friend", a = "Привет мой друг"},
    {q = "Good morning Sir", a = "Доброе утро Сэр"},
    {q = "Thank you very", a = "Спасибо тебе большое"},
    {q = "Have a nice", a = "Хорошего тебе дня"},
    
    -- Кто я и кто ты
    {q = "My name is", a = "Мое имя есть"},
    {q = "I am a", a = "Я есть этот"},
    {q = "Who are you", a = "Кто есть ты"},
    {q = "You are hero", a = "Ты есть герой"},
    {q = "He is Lua", a = "Он есть Луа"},
    
    -- Геймерские команды (с большой буквы)
    {q = "Kill the Boss", a = "Убей этого Босса"},
    {q = "Follow the leader", a = "Следуй за лидером"},
    {q = "Protect our base", a = "Защищай нашу базу"},
    {q = "Enemy is here", a = "Враг есть здесь"},
    {q = "Shoot that target", a = "Стреляй в цель"},
    {q = "Go to Mid", a = "Иди на Мид"},
    
    -- Простые действия
    {q = "I see Moon", a = "Я вижу Луну"},
    {q = "We love games", a = "Мы любим игры"},
    {q = "They play football", a = "Они играют футбол"},
    {q = "Open the door", a = "Открой эту дверь"},
    {q = "Close the window", a = "Закрой это окно"},
    
    -- Описание мира
    {q = "Earth is round", a = "Земля есть круглая"},
    {q = "Sky is blue", a = "Небо есть синее"},
    {q = "Fire is hot", a = "Огонь есть горячий"},
    {q = "Ice is cold", a = "Лед есть холодный"},
    {q = "Sun shines bright", a = "Солнце светит ярко"},
    
    -- Вопросы
    {q = "What is this", a = "Что это такое"},
    {q = "Where is home", a = "Где есть дом"},
    {q = "When is start", a = "Когда есть начало"},
    {q = "Why you run", a = "Почему ты бежишь"},
    {q = "How to craft", a = "Как это крафтить"},
    
    -- Работа и Техника
    {q = "Laptop is new", a = "Ноутбук есть новый"},
    {q = "Phone is broken", a = "Телефон есть сломан"},
    {q = "Code is working", a = "Код есть работает"},
    {q = "Internet is fast", a = "Интернет есть быстрый"},
    {q = "Battery is low", a = "Батарея есть низкая"},
    
    -- Разное
    {q = "I have dream", a = "Я имею мечту"},
    {q = "Life is beautiful", a = "Жизнь есть прекрасна"},
    {q = "Peace is good", a = "Мир это хорошо"},
    {q = "Time is money", a = "Время это деньги"},
    {q = "Stop it now", a = "Останови это сейчас"},
    {q = "Wait for me", a = "Подожди меня здесь"},
    {q = "Be very careful", a = "Будь очень осторожен"},
    {q = "God is great", a = "Бог есть велик"},
    {q = "Never give up", a = "Никогда не сдавайся"},
    {q = "Run away now", a = "Беги прочь сейчас"},
    {q = "Look at top", a = "Смотри на верх"},
    {q = "Check the map", a = "Проверь эту карту"},
    {q = "Win the match", a = "Выиграй этот матч"},
	
	{q = "Hello how are", a = "Привет как твои"},
    {q = "I am fine", a = "Я в порядке"},
    {q = "What is name", a = "Как твое имя"},
    {q = "My name Lua", a = "Мое имя Луа"},
    {q = "Nice to meet", a = "Приятно познакомиться снова"},
    {q = "How old you", a = "Сколько тебе лет"},
    {q = "I am young", a = "Я еще молодой"},
    {q = "Where you live", a = "Где ты живешь"},
    {q = "I live here", a = "Я живу здесь"},
    {q = "Who is this", a = "Кто это такой"},

    -- Повседневные вопросы
    {q = "What you doing", a = "Что ты делаешь"},
    {q = "I am working", a = "Я сейчас работаю"},
    {q = "Are you hungry", a = "Ты хочешь есть"},
    {q = "Yes I am", a = "Да я хочу"},
    {q = "No I not", a = "Нет я не"},
    {q = "Do you know", a = "Ты это знаешь"},
    {q = "I dont know", a = "Я не знаю"},
    {q = "Please help me", a = "Пожалуйста помоги мне"},
    {q = "Wait a minute", a = "Подожди одну минуту"},
    {q = "Come with me", a = "Пойдем со мной"},

    -- Мнения и чувства
    {q = "I like it", a = "Мне это нравится"},
    {q = "I hate this", a = "Я это ненавижу"},
    {q = "It is good", a = "Это очень хорошо"},
    {q = "It is bad", a = "Это очень плохо"},
    {q = "Are you sure", a = "Ты в этом уверен"},
    {q = "I am sure", a = "Я в этом уверен"},
    {q = "What you think", a = "Что ты думаешь"},
    {q = "I think so", a = "Я так думаю"},
    {q = "You are right", a = "Ты совершенно прав"},
    {q = "You are wrong", a = "Ты не прав"},

    -- Место и время
    {q = "Where is it", a = "Где это находится"},
    {q = "It is there", a = "Это находится там"},
    {q = "What time now", a = "Сколько сейчас времени"},
    {q = "It is late", a = "Уже очень поздно"},
    {q = "I must go", a = "Мне пора идти"},
    {q = "When you come", a = "Когда ты придешь"},
    {q = "I come tomorrow", a = "Я приду завтра"},
    {q = "How much cost", a = "Сколько это стоит"},
    {q = "It is cheap", a = "Это стоит дешево"},
    {q = "It is expensive", a = "Это стоит дорого"},

    -- Прощание
    {q = "See you later", a = "Увидимся с тобой"},
    {q = "Have a luck", a = "Желаю тебе удачи"},
    {q = "Good night friend", a = "Спокойной ночи друг"},
    {q = "Bye bye now", a = "Пока пока тебе"},
    {q = "Take your time", a = "Не нужно спешить"},
    {q = "Be very careful", a = "Будь очень осторожен"},
    {q = "Tell me more", a = "Расскажи мне больше"},
    {q = "I am listening", a = "Я тебя слушаю"},
    {q = "Stop talking now", a = "Хватит сейчас болтать"},
    {q = "Everything is fine", a = "Все будет хорошо"},
	
	-- Планы и намерения
    {q = "I want to go for a walk today", a = "Я хочу пойти на прогулку сегодня"},
    {q = "What are you going to do tomorrow", a = "Что ты собираешься делать завтра"},
    {q = "I think we should go to city", a = "Я думаю нам стоит пойти в город"},
    {q = "Do you want to play with us", a = "Ты хочешь поиграть вместе с нами"},
    {q = "I am going to sleep very soon", a = "Я собираюсь лечь спать очень скоро"},
    {q = "Tell me about your new car now", a = "Расскажи мне о своей новой машине"},

    -- Работа и учёба
    {q = "I have a lot of work today", a = "У меня сегодня очень много работы"},
    {q = "Can you explain this rule to me", a = "Можешь ты объяснить мне это правило"},
    {q = "I need to learn more Lua code", a = "Мне нужно учить больше кода Луа"},
    {q = "This task is very hard for me", a = "Это задание очень трудное для меня"},
    {q = "When will you finish your big project", a = "Когда ты закончишь свой большой проект"},
    {q = "I am looking for a new job", a = "Я сейчас ищу для себя новую работу"},

    -- Здоровье и чувства
    {q = "I feel very tired after this work", a = "Я чувствую себя усталым после работы"},
    {q = "Are you feeling better than yesterday morning", a = "Ты чувствуешь себя лучше чем вчера утром"},
    {q = "I think I should see a doctor", a = "Я думаю мне стоит показаться врачу"},
    {q = "I am really happy to see you", a = "Я действительно очень рад тебя видеть"},
    {q = "Don't worry everything will be just fine", a = "Не волнуйся все будет просто хорошо"},

    -- Покупки и услуги
    {q = "How much does this new phone cost", a = "Сколько стоит этот новый мобильный телефон"},
    {q = "I would like to buy a ticket", a = "Я хотел бы купить один билет"},
    {q = "Can you show me the way there", a = "Можете показать мне дорогу туда"},
    {q = "I am looking for the nearest shop", a = "Я ищу самый ближайший магазин"},
    {q = "Do you accept credit cards here now", a = "Вы принимаете здесь кредитные карты сейчас"},

    -- Мнения и обсуждения
    {q = "I don't think this is a good idea", a = "Я не думаю что это хорошая идея"},
    {q = "What do you think about this book", a = "Что ты думаешь об этой книге"},
    {q = "It was the best game in life", a = "Это была лучшая игра в жизни"},
    {q = "I agree with you on this point", a = "Я согласен с тобой в этом вопросе"},
    {q = "You are absolutely right about that thing", a = "Ты совершенно прав насчет той вещи"},

    -- Сложные просьбы
    {q = "Could you please open the window now", a = "Не мог бы ты открыть окно сейчас"},
    {q = "Please tell me the truth about it", a = "Пожалуйста скажи мне правду об этом"},
    {q = "Can you help me with my bag", a = "Можешь ты помочь мне с сумкой"},
    {q = "Wait for me near the old house", a = "Подожди меня возле того старого дома"},
    {q = "Don't forget to call me later today", a = "Не забудь позвонить мне сегодня позже"},

    -- Время и место
    {q = "The weather is too cold for walk", a = "Погода слишком холодная для прогулки"},
    {q = "It was a very long and hard day", a = "Это был очень длинный и тяжелый день"},
    {q = "I will be there in ten minutes", a = "Я буду там через десять минут"},
    {q = "Where did you find this beautiful flower", a = "Где ты нашел этот прекрасный цветок"},
    {q = "I lived in this city for years", a = "Я жил в этом городе годами"},

    -- Разное (бытовое)
    {q = "I forgot where I put my keys", a = "Я забыл куда я положил ключи"},
    {q = "Turn off the light and go sleep", a = "Выключи свет и иди уже спать"},
    {q = "We are going to have a dinner", a = "Мы собираемся пойти на ужин"},
    {q = "I like to listen to loud music", a = "Мне нравится слушать очень громкую музыку"},
    {q = "Can you speak a bit more slow", a = "Можешь ты говорить немного более медленно"},
    {q = "I don't understand what you are saying", a = "Я не понимаю что ты сейчас говоришь"},
    {q = "Give me a chance to explain it", a = "Дай мне шанс это все объяснить"},
    {q = "It is time to start our lesson", a = "Пришло время начинать наш урок"},
    {q = "I hope you will win the match", a = "Я надеюсь ты выиграешь этот матч"},
    {q = "Stay with me for a little while", a = "Останься со мной еще на чуть чуть"},
    {q = "The sky looks very dark today morning", a = "Небо выглядит очень темным сегодня утром"},
    {q = "I am so proud of your success", a = "Я так горжусь твоим большим успехом"},
    {q = "Let us make this world much better", a = "Давайте сделаем этот мир гораздо лучше"},
	
	-- Технологии и Будущее
    {q = "Artificial intelligence will change the way we live and work forever", a = "Искусственный интеллект изменит то как мы живем и работаем навсегда"},
    {q = "Modern technology makes our lives much easier and more comfortable today", a = "Современные технологии делают нашу жизнь гораздо проще и комфортнее сегодня"},
    {q = "We need to develop new sources of clean and safe energy", a = "Нам нужно развивать новые источники чистой и безопасной энергии сейчас"},
    {q = "Virtual reality can help us to learn and explore new worlds", a = "Виртуальная реальность может помочь нам учиться и исследовать новые миры"},
    {q = "Computers are becoming smarter and more powerful every single year now", a = "Компьютеры становятся умнее и мощнее каждый божий год прямо сейчас"},

    -- Философия и Жизнь
    {q = "Knowledge is the only thing that can make us truly free", a = "Знание это единственная вещь которая может сделать нас поистине свободными"},
    {q = "Sometimes we have to make very difficult choices in our lives", a = "Иногда нам приходится делать очень трудный выбор в нашей жизни"},
    {q = "Success depends on how hard you are willing to work today", a = "Успех зависит от того как усердно ты готов работать сегодня"},
    {q = "True happiness comes from within and not from the outside world", a = "Истинное счастье приходит изнутри а не из этого внешнего мира"},
    {q = "Every person has the power to change the whole world around", a = "Каждый человек имеет силу изменить весь мир вокруг себя сейчас"},

    -- Бизнес и Переговоры
    {q = "We should discuss the terms of our new contract very carefully", a = "Нам следует обсудить условия нашего нового контракта очень внимательно"},
    {q = "The company is looking for a professional with a lot of experience", a = "Компания ищет профессионала с очень большим опытом работы прямо сейчас"},
    {q = "We need to find a way to reduce our monthly expenses", a = "Нам нужно найти способ сократить наши ежемесячные расходы в компании"},
    {q = "Innovation is the key to success in this modern global market", a = "Инновации это ключ к успеху на этом современном глобальном рынке"},
    {q = "I would like to schedule a meeting for the next Monday", a = "Я хотел бы назначить встречу на следующий понедельник утром сейчас"},

    -- Наука и Природа
    {q = "Global warming is a serious problem for the future of humanity", a = "Глобальное потепление это серьезная проблема для будущего всего человечества"},
    {q = "Scientists are working on a new medicine for this dangerous disease", a = "Ученые работают над новым лекарством от этой очень опасной болезни"},
    {q = "Space exploration gives us a better understanding of the whole universe", a = "Исследование космоса дает нам лучшее понимание всей этой вселенной"},
    {q = "Protecting the environment is our responsibility to the next young generations", a = "Защита окружающей среды это наша ответственность перед следующими молодыми поколениями"},
    {q = "The stars shine brightly in the dark and silent night sky", a = "Звезды светят ярко в этом темном и тихом ночном небе"},

    -- Сложные условия (Если / Потому что)
    {q = "If you study hard you will definitely achieve your main goal", a = "Если ты будешь учиться усердно ты обязательно достигнешь своей цели"},
    {q = "I cannot go with you because I have too much work", a = "Я не могу пойти с тобой потому что имею много работы"},
    {q = "Even if it rains we will still go to the forest", a = "Даже если пойдет дождь мы все равно пойдем в этот лес"},
    {q = "Although he was tired he continued to work on his project", a = "Хотя он был усталым он продолжал работать над своим проектом"},
    {q = "She decided to stay at home because she felt very sick", a = "Она решила остаться дома потому что она чувствовала себя больной"},

    -- Психология и Отношения
    {q = "Communication is the foundation of any strong and healthy long relationship", a = "Общение это фундамент любых сильных и здоровых долгих отношений"},
    {q = "We should respect each other and try to understand different points", a = "Мы должны уважать друг друга и пытаться понимать разные точки"},
    {q = "Do not be afraid of making mistakes because they make you", a = "Не бойся совершать ошибки потому что они делают тебя только лучше"},
    {q = "Trust is very hard to build but very easy to break", a = "Доверие очень трудно построить но очень легко его можно сломать"},
    {q = "I am very grateful for your support during this difficult time", a = "Я очень благодарен за твою поддержку в это трудное время"},

    -- Описательные (Повествование)
    {q = "He was walking alone down the street when the rain started", a = "Он гулял один по этой улице когда начался этот сильный дождь"},
    {q = "The old house was standing at the edge of the forest", a = "Старый дом стоял на самом краю этого очень густого леса"},
    {q = "She opened the book and started to read the first page", a = "Она открыла книгу и начала читать самую первую страницу текста"},
    {q = "They traveled across the ocean to find a new beautiful home", a = "Они путешествовали через океан чтобы найти новый прекрасный дом себе"},
    {q = "The sun was slowly setting behind the high and blue mountains", a = "Солнце медленно заходило за эти высокие и очень синие горы"},

    -- Социальные темы
    {q = "Education should be available to every child in the whole world", a = "Образование должно быть доступно каждому ребенку во всем этом мире"},
    {q = "We need to fight for our rights and for our freedom", a = "Нам нужно сражаться за наши права и за нашу свободу"},
    {q = "Helping others is a noble thing that makes us much better", a = "Помощь другим это благородное дело которое делает нас гораздо лучше"},
    {q = "Music has the power to bring people from different cultures together", a = "Музыка имеет силу объединять людей из разных культур вместе сегодня"},
    {q = "Peace is the most important thing for the progress of humanity", a = "Мир это самая важная вещь для прогресса всего нашего человечества"},
	
	-- Погода и окружение (Без лишнего "есть")
    {q = "The sun is bright", a = "Солнце ярко светит"},
    {q = "The sky is blue", a = "Небо сегодня синее"},
    {q = "It is raining now", a = "Сейчас идет дождь"},
    {q = "It is snowing now", a = "Сейчас идет снег"},
    {q = "The wind is strong", a = "Дует сильный ветер"},
    {q = "It is very cold", a = "На улице очень холодно"},
    {q = "It is very hot", a = "Сегодня очень жарко"},
    
    -- Описание предметов и состояний
    {q = "The cat is here", a = "Кошка сидит здесь"},
    {q = "The dog is there", a = "Собака бегает там"},
    {q = "The door is open", a = "Дверь сейчас открыта"},
    {q = "The window is closed", a = "Окно плотно закрыто"},
    {q = "This book is interesting", a = "Эта книга интересная"},
    {q = "The coffee is cold", a = "Кофе уже остыл"},
    {q = "My tea is sweet", a = "Мой чай сладкий"},

    -- Ежедневные действия (Живая речь)
    {q = "I wash my face", a = "Я умываю лицо"},
    {q = "I brush my teeth", a = "Я чищу зубы"},
    {q = "I drink hot coffee", a = "Я пью горячий кофе"},
    {q = "I eat my breakfast", a = "Я завтракаю сейчас"},
    {q = "I go to work", a = "Я иду на работу"},
    {q = "I study every day", a = "Я учусь каждый день"},
    {q = "I read the news", a = "Я читаю новости"},
    {q = "I want to sleep", a = "Я хочу спать"},

    -- Чувства и люди
    {q = "I am at home", a = "Я сейчас дома"},
    {q = "You are at school", a = "Ты сейчас в школе"},
    {q = "He is in city", a = "Он сейчас в городе"},
    {q = "She is very happy", a = "Она очень счастлива"},
    {q = "They are very tired", a = "Они очень устали"},
    {q = "We are all together", a = "Мы все вместе"},
    {q = "I am very hungry", a = "Я очень проголодался"},
    {q = "I am so tired", a = "Я так сильно устал"},

    -- Короткие фразы и просьбы
    {q = "Give me that pen", a = "Дай мне ту ручку"},
    {q = "Look at the sky", a = "Посмотри на небо"},
    {q = "Tell me the story", a = "Расскажи мне историю"},
    {q = "Help me with this", a = "Помоги мне с этим"},
    {q = "Wait for me here", a = "Подожди меня здесь"},
    {q = "Call your best friend", a = "Позвони лучшему другу"},
    {q = "Open the big door", a = "Открой большую дверь"},
    {q = "Write your full name", a = "Напиши свое имя"},

    -- Магазин и покупки
    {q = "How much is it", a = "Сколько это стоит"},
    {q = "I have no money", a = "У меня нет денег"},
    {q = "I want an apple", a = "Я хочу яблоко"},
    {q = "Buy some fresh bread", a = "Купи свежего хлеба"},
    {q = "Where is the shop", a = "Где находится магазин"},
    {q = "I need some help", a = "Мне нужна помощь"},

    -- Время и планы
    {q = "Today is a holiday", a = "Сегодня выходной день"},
    {q = "Tomorrow is my birthday", a = "Завтра мой день рождения"},
    {q = "The week is over", a = "Неделя подошла к концу"},
    {q = "It is very late", a = "Сейчас уже поздно"},
    {q = "I must go now", a = "Мне пора идти"},
    {q = "See you later today", a = "Увидимся сегодня позже"},
	
    -- Быт и дом
    {q = "I lost my keys", a = "Я потерял ключи"},
    {q = "The coffee is too hot", a = "Кофе слишком горячий"},
    {q = "Turn off the TV", a = "Выключи этот телевизор"},
    {q = "I have a question", a = "У меня возник вопрос"},
    {q = "Show me the way", a = "Покажи мне дорогу"},
    {q = "The soup is salty", a = "Этот суп пересолен"},
    {q = "Clean your own room", a = "Уберись в своей комнате"},
    {q = "The light is off", a = "Свет сейчас выключен"},
    {q = "Where is my phone", a = "Где мой телефон"},
    {q = "The bed is soft", a = "Кровать очень мягкая"},

    -- Работа и дела
    {q = "I am very busy", a = "Я очень сильно занят"},
    {q = "He is my boss", a = "Он мой начальник"},
    {q = "The meeting is over", a = "Собрание уже закончилось"},
    {q = "We need more time", a = "Нам нужно больше времени"},
    {q = "Sign this important paper", a = "Подпиши эту важную бумагу"},
    {q = "The project is ready", a = "Проект полностью готов"},
    {q = "I have a job", a = "У меня есть работа"},
    {q = "Send me the file", a = "Пришли мне этот файл"},
    {q = "The office is closed", a = "Офис уже закрыт"},
    {q = "Check your email now", a = "Проверь свою почту"},

    -- Эмоции и люди
    {q = "I am proud of you", a = "Я горжусь тобой"},
    {q = "She is my sister", a = "Она моя сестра"},
    {q = "He is a doctor", a = "Он работает врачом"},
    {q = "They are my friends", a = "Они мои друзья"},
    {q = "I am so happy", a = "Я так счастлив"},
    {q = "Don't be so sad", a = "Не грусти так сильно"},
    {q = "Are you very angry", a = "Ты сильно злишься"},
    {q = "I believe in you", a = "Я верю в тебя"},
    {q = "We are very tired", a = "Мы очень сильно устали"},
    {q = "Tell me the truth", a = "Скажи мне правду"},

    -- Город и транспорт
    {q = "Where is the bus", a = "Где стоит автобус"},
    {q = "The train is coming", a = "Поезд уже идет"},
    {q = "I need a taxi", a = "Мне нужно такси"},
    {q = "The car is fast", a = "Машина едет быстро"},
    {q = "Stop the car here", a = "Останови машину здесь"},
    {q = "The street is noisy", a = "На улице шумно"},
    {q = "The shop is far", a = "Магазин находится далеко"},
    {q = "Let's cross the road", a = "Давай перейдем дорогу"},
    {q = "I am at station", a = "Я нахожусь на вокзале"},
    {q = "The map is wrong", a = "Карта показывает неверно"},

    -- Разное (природа, время, детали)
    {q = "The sky looks gray", a = "Небо кажется серым"},
    {q = "It is getting dark", a = "На улице темнеет"},
    {q = "The flowers are beautiful", a = "Цветы очень красивые"},
    {q = "I like this music", a = "Мне нравится эта музыка"},
    {q = "The movie was boring", a = "Фильм был скучным"},
    {q = "Time is running out", a = "Время на исходе"},
    {q = "I forgot the name", a = "Я забыл имя"},
    {q = "Look at the stars", a = "Посмотри на звезды"},
    {q = "The water is cold", a = "Вода очень холодная"},
    {q = "Everything will be fine", a = "Все будет хорошо"},
	
	-- Бой и Команды (Шутеры)
    {q = "Enemy spotted here", a = "Заметил врага здесь"},
    {q = "Cover me now", a = "Прикрой меня сейчас"},
    {q = "Fire in hole", a = "Ложись сейчас граната"},
    {q = "Shoot that guy", a = "Стреляй в него"},
    {q = "Hold this point", a = "Держи эту точку"},
    {q = "Rush the site", a = "Быстро бежим туда"},
    {q = "Need backup here", a = "Нужна помощь здесь"},
    {q = "Reloading my gun", a = "Перезаряжаю свое оружие"},
    {q = "Nice shot bro", a = "Хороший выстрел бро"},
    {q = "One shot left", a = "Остался один выстрел"},

    -- RPG и MMO (Прокачка, Боссы)
    {q = "Level up fast", a = "Быстро поднял уровень"},
    {q = "Boss is defeated", a = "Босс успешно повержен"},
    {q = "Loot the chest", a = "Обыщи этот сундук"},
    {q = "Mana is low", a = "Мана почти закончилась"},
    {q = "Heal me please", a = "Полечи меня пожалуйста"},
    {q = "Equip the sword", a = "Возьми этот меч"},
    {q = "Quest is finished", a = "Задание уже выполнено"},
    {q = "Find the NPC", a = "Найди этого персонажа"},
    {q = "Skill is ready", a = "Навык готов сейчас"},
    {q = "Craft new armor", a = "Скрафти новую броню"},

    -- Чат и Общение
    {q = "Good luck guys", a = "Удачи всем ребятам"},
    {q = "Dont be toxic", a = "Не будь токсичным"},
    {q = "Mute that player", a = "Заглуши этого игрока"},
    {q = "Report this cheater", a = "Кинь жалобу читеру"},
    {q = "Wait for me", a = "Подожди меня здесь"},
    {q = "Follow the leader", a = "Следуй за лидером"},
    {q = "Go to mid", a = "Иди на мид"},
    {q = "Stay at base", a = "Оставайся на базе"},
    {q = "I am lagging", a = "У меня лагает"},
    {q = "GG well played", a = "ГГ хорошо сыграли"},

    -- Технические проблемы и Состояние
    {q = "Ping is high", a = "Пинг слишком высокий"},
    {q = "Server is full", a = "Сервер сейчас полон"},
    {q = "Update the game", a = "Обнови эту игру"},
    {q = "Save your progress", a = "Сохрани свой прогресс"},
    {q = "Inventory is full", a = "Инвентарь забит полностью"},
    {q = "Buy the skin", a = "Купи этот скин"},
    {q = "Trade with me", a = "Торгуй со мной"},
    {q = "Find the exit", a = "Найди выход отсюда"},
    {q = "Game is over", a = "Игра уже окончена"},
    {q = "Play one more", a = "Сыграем еще разок"},

    -- Разное (Геймплей)
    {q = "Hide in bush", a = "Прячься в кустах"},
    {q = "Jump over wall", a = "Прыгай через стену"},
    {q = "Run away now", a = "Беги прочь сейчас"},
    {q = "Look at map", a = "Посмотри на карту"},
    {q = "Use the potion", a = "Используй это зелье"},
    {q = "Build the wall", a = "Построй эту стену"},
    {q = "Stop the raid", a = "Останови этот рейд"},
    {q = "Open the gate", a = "Открой эти ворота"},
    {q = "Victory is ours", a = "Победа за нами"},
    {q = "Defeat is close", a = "Поражение совсем близко"},
	
	-- Ресурсы и Добыча
    {q = "I need some wood", a = "Мне нужно немного дерева"},
    {q = "Mine the stone blocks", a = "Добывай эти каменные блоки"},
    {q = "Collect some iron ore", a = "Собери немного железной руды"},
    {q = "Find some rare diamonds", a = "Найди эти редкие алмазы"},
    {q = "Chop the tall trees", a = "Руби эти высокие деревья"},
    {q = "Dig the dirt here", a = "Копай землю прямо здесь"},
    {q = "Look for some coal", a = "Поищи немного этого угля"},
    {q = "I found some gold", a = "Я нашел немного золота"},

    -- Крафт и Инструменты
    {q = "Use the wooden workbench", a = "Используй этот деревянный верстак"},
    {q = "Craft a new axe", a = "Скрафти себе новый топор"},
    {q = "Make a stone pickaxe", a = "Сделай одну каменную кирку"},
    {q = "Repair my iron sword", a = "Почини мой железный меч"},
    {q = "Upgrade your old tools", a = "Улучши свои старые инструменты"},
    {q = "Melt the iron ore", a = "Переплавь эту железную руду"},
    {q = "Smelt the gold sand", a = "Переплавь этот золотой песок"},
    {q = "Craft a hunting bow", a = "Скрафти один охотничий лук"},
    {q = "Arrows are very expensive", a = "Стрелы стоят очень дорого"},

    -- Строительство и База
    {q = "Build a wooden wall", a = "Построй одну деревянную стену"},
    {q = "Place the stone block", a = "Поставь этот каменный блок"},
    {q = "Put items in chest", a = "Положи вещи в сундук"},
    {q = "Fix the broken roof", a = "Чини эту сломанную крышу"},
    {q = "Open the heavy door", a = "Открой эту тяжелую дверь"},
    {q = "Light up the house", a = "Освети этот дом сейчас"},
    {q = "Break the stone wall", a = "Сломай эту каменную стену"},
    {q = "Craft a warm bed", a = "Скрафти одну теплую кровать"},
    {q = "Set a spawn point", a = "Поставь точку своего спавна"},

    -- Выживание и Голод
    {q = "I am very hungry", a = "Я очень сильно проголодался"},
    {q = "Eat some cooked meat", a = "Съешь это жареное мясо"},
    {q = "Drink some clean water", a = "Попей этой чистой воды"},
    {q = "Cook the raw fish", a = "Пожарь эту сырую рыбу"},
    {q = "Plant the wheat seeds", a = "Посади эти семена пшеницы"},
    {q = "Collect some wild berries", a = "Собери эти лесные ягоды"},
    {q = "Check your hunger bar", a = "Проверь свою полоску голода"},

    -- Опасность и Ночь
    {q = "Night is coming soon", a = "Скоро наступит темная ночь"},
    {q = "It is too dark", a = "Здесь сейчас слишком темно"},
    {q = "Zombies are outside now", a = "Зомби сейчас бродят снаружи"},
    {q = "Run to the base", a = "Беги скорее на базу"},
    {q = "Stay inside the house", a = "Оставайся внутри этого дома"},
    {q = "Hide from the monsters", a = "Прячься от этих монстров"},
    {q = "Find a safe place", a = "Найди одно безопасное место"},
    {q = "Explore the deep cave", a = "Исследуй эту глубокую пещеру"},

    -- ПвП и Рейды (Rust стиль)
    {q = "Raid the enemy base", a = "Рейди базу этого врага"},
    {q = "Protect our loot now", a = "Защищай наш лут сейчас"},
    {q = "Share some food please", a = "Поделись едой пожалуйста"},
    {q = "Give me some ammo", a = "Дай мне немного патронов"},
    {q = "He has a gun", a = "У него есть пушка"},
    {q = "They are attacking us", a = "Они сейчас нападают нас"},
    {q = "Who is raiding us", a = "Кто сейчас рейдит нас"},
    {q = "Use the sharp knife", a = "Используй этот острый нож"},
    {q = "Wear the iron armor", a = "Надень эту железную броню"},
	
	-- Продвинутый крафт и механизмы
    {q = "Put the coal in furnace", a = "Положи этот уголь в печь"},
    {q = "Use the heavy anvil now", a = "Используй эту тяжелую наковальню"},
    {q = "Craft a metal window frame", a = "Скрафти одну металлическую раму"},
    {q = "Make some clear glass blocks", a = "Сделай несколько чистых стекол"},
    {q = "Repair the damaged workbench", a = "Почини этот поврежденный верстак"},
    {q = "Melt the scrap metal now", a = "Переплавь этот металлолом сейчас"},
    {q = "I need a blueprint first", a = "Мне сначала нужен этот чертеж"},
    {q = "Learn a new crafting recipe", a = "Выучи один новый рецепт крафта"},

    -- Фермерство и Животные
    {q = "Use the stone hoe here", a = "Используй здесь каменную мотыгу"},
    {q = "Harvest the golden wheat", a = "Собери эту золотую пшеницу"},
    {q = "Plant the corn seeds now", a = "Посади семена кукурузы сейчас"},
    {q = "Shear the white sheep", a = "Постриги эту белую овцу"},
    {q = "Feed the small pig please", a = "Покорми эту маленькую свинью пожалуйста"},
    {q = "Milk the brown cow now", a = "Подои эту коричневую корову сейчас"},
    {q = "Water the dry plants", a = "Полей эти сухие растения"},

    -- Защита базы и Ловушки
    {q = "Build a high stone tower", a = "Построй одну высокую каменную башню"},
    {q = "Set a hidden trap here", a = "Поставь здесь одну скрытую ловушку"},
    {q = "Build a strong wood fence", a = "Построй один крепкий деревянный забор"},
    {q = "Light the bright torch now", a = "Зажги этот яркий факел сейчас"},
    {q = "Check the base lock now", a = "Проверь замок на базе сейчас"},
    {q = "Upgrade the walls to stone", a = "Улучши эти стены до камня"},
    {q = "The base is decaying fast", a = "Эта база сейчас быстро разрушается"},

    -- Состояния персонажа (Rust/DayZ стиль)
    {q = "I have radiation poisoning", a = "У меня сейчас радиационное отравление"},
    {q = "Stop the heavy bleeding now", a = "Останови это сильное кровотечение сейчас"},
    {q = "I feel very cold here", a = "Я чувствую сильный холод здесь"},
    {q = "I need a medical kit", a = "Мне нужна одна аптечка"},
    {q = "Drink the clean water now", a = "Попей эту чистую воду сейчас"},
    {q = "Use the blue bandage now", a = "Используй этот синий бинт сейчас"},
    {q = "My stamina is very low", a = "Моя выносливость сейчас очень низкая"},

    -- Движение и Действия
    {q = "Crouch and stay very quiet", a = "Присядь и веди себя тихо"},
    {q = "Climb up the long ladder", a = "Забирайся вверх по этой лестнице"},
    {q = "Swim across the big lake", a = "Переплыви через это большое озеро"},
    {q = "Jump down from the roof", a = "Прыгай вниз прямо с крыши"},
    {q = "Run fast to the forest", a = "Беги быстрее в этот лес"},
    {q = "Sneak into the enemy base", a = "Прокрадись на базу этого врага"},

    -- Ресурсы (Продвинутые)
    {q = "Collect some sulfur ore blocks", a = "Собери немного блоков серной руды"},
    {q = "Find some metal scrap parts", a = "Найди немного металлических запчастей"},
    {q = "I found some gunpowder here", a = "Я нашел немного пороха здесь"},
    {q = "Need more low grade fuel", a = "Нужно больше этого дешевого топлива"},
    {q = "Gather some white wool now", a = "Собери немного белой шерсти сейчас"},
    {q = "Look for the red berries", a = "Поищи эти красные ягоды"},

    -- Общее
    {q = "Share the loot with team", a = "Поделись лутом со своей командой"},
    {q = "Wait for the sunrise here", a = "Подожди здесь этого восхода солнца"},
    {q = "The map is very big", a = "Эта карта сейчас очень большая"},
    {q = "Find the hidden cave entrance", a = "Найди этот скрытый вход пещеру"},
    {q = "Is anyone near the base", a = "Есть кто нибудь возле базы"},
    {q = "Watch out for the bears", a = "Остерегайся этих диких медведей"},
    {q = "I am lost in forest", a = "Я потерялся в этом лесу"},
    {q = "Craft a sleeping bag now", a = "Скрафти один спальный мешок сейчас"},
    {q = "The server is restarting soon", a = "Сервер скоро будет перезагружен"},
	
	-- Приветствия и повседневное
    {q = "Hey, how is it going", a = "Эй, как твои дела"},
    {q = "I haven't seen you for ages", a = "Я не видел тебя целую вечность"},
    {q = "It is nice to see you", a = "Рад тебя видеть снова"},
    {q = "What have you been up to", a = "Чем ты занимался все это время"},
    {q = "Nothing much, just working a lot", a = "Ничего особенного, просто много работаю"},
    {q = "I am so glad you came", a = "Я так рад, что ты пришел"},

    -- Мнения и обсуждения
    {q = "What do you think about that", a = "Что ты думаешь об этом"},
    {q = "To be honest, I don't know", a = "Честно говоря, я не знаю"},
    {q = "That sounds like a great plan", a = "Это звучит как отличный план"},
    {q = "I am not sure about this", a = "Я не уверен насчет этого"},
    {q = "You are absolutely right, my friend", a = "Ты совершенно прав, мой друг"},
    {q = "I don't really care about it", a = "Мне на самом деле все равно"},

    -- Просьбы и помощь
    {q = "Could you do me a favor", a = "Можешь сделать мне одно одолжение"},
    {q = "I need your help with this", a = "Мне нужна твоя помощь с этим"},
    {q = "Can you explain this to me", a = "Можешь мне это объяснить"},
    {q = "Don't worry, I will help you", a = "Не волнуйся, я тебе помогу"},
    {q = "Give me a second, please", a = "Дай мне секунду, пожалуйста"},
    {q = "Wait for me outside the house", a = "Подожди меня на улице возле дома"},

    -- Эмоции и реакция
    {q = "I am so sorry to hear", a = "Мне так жаль это слышать"},
    {q = "Are you kidding me right now", a = "Ты сейчас шутишь надо мной"},
    {q = "That is totally unbelievable, wow", a = "Это просто невероятно, вау"},
    {q = "I am really proud of you", a = "Я действительно горжусь тобой"},
    {q = "Don't be mad at me, okay", a = "Не злись на меня, хорошо"},
    {q = "Everything will be just fine, relax", a = "Все будет хорошо, расслабься"},

    -- Уточнения и вопросы
    {q = "What do you mean by that", a = "Что ты имеешь в виду"},
    {q = "Can you repeat that again, please", a = "Можешь повторить это еще раз"},
    {q = "Are you busy at the moment", a = "Ты сейчас сильно занят"},
    {q = "Is everything okay with you", a = "У тебя все в порядке"},
    {q = "What happened to your old car", a = "Что случилось с твоей старой машиной"},

    -- Планы и время
    {q = "Let's meet tomorrow near the park", a = "Давай встретимся завтра возле парка"},
    {q = "I will be back in ten", a = "Я вернусь через десять минут"},
    {q = "It is time to go home", a = "Пришло время идти домой"},
    {q = "I have to leave right now", a = "Мне нужно уйти прямо сейчас"},
    {q = "See you later this evening, bye", a = "Увидимся сегодня вечером, пока"},

    -- Бытовое и разное
    {q = "I forgot what I wanted to say", a = "Я забыл, что хотел сказать"},
    {q = "Tell me the whole story now", a = "Расскажи мне всю историю сейчас"},
    {q = "I am looking for my phone", a = "Я ищу свой мобильный телефон"},
    {q = "Do you have any spare money", a = "У тебя есть лишние деньги"},
    {q = "It doesn't matter anymore, forget it", a = "Это больше не важно, забудь"},
    {q = "Stop telling me what to do", a = "Хватит говорить мне, что делать"},

    -- Геймерский живой диалог
    {q = "Do you want to play together", a = "Ты хочешь поиграть вместе"},
    {q = "I am tired of this game", a = "Я устал от этой игры"},
    {q = "That was a really close match", a = "Это был очень потный матч"},
    {q = "We almost won that last round", a = "Мы почти выиграли прошлый раунд"},
    {q = "I will carry you this time", a = "Я затащу тебя в этот раз"},
    {q = "Don't forget to save the game", a = "Не забудь сохранить игру"},
    {q = "I am laggy, wait for me", a = "У меня лаги, подожди меня"},
    {q = "Who is the leader of team", a = "Кто лидер в этой команде"},
    {q = "Good luck and have fun, guys", a = "Удачи и веселой игры, ребята"},
    {q = "Take care of yourself, my friend", a = "Береги себя, мой дорогой друг"}
}

-- Обучаем словарь на всех текстах (и вопросах, и ответах)
print("Сбор слов для словаря...")
for _, pair in ipairs(raw_data) do
    ai:fit(pair.q)
    ai:fit(pair.a)
end
print("Словарь готов. Всего слов: " .. ai.next_id)

-- 3. Создаем модель в DJL
print("Создание модели " .. model_id .. "...")
ai:create()

-- 4. Кодируем текстовые данные в числа для нейросети
local train_data = {
    inputs = {},
    labels = {}
}

for _, pair in ipairs(raw_data) do
    -- Превращаем текст в таблицы чисел нужной длины
    table.insert(train_data.inputs, ai:encode(pair.q, IN_LEN))
    table.insert(train_data.labels, ai:encode(pair.a, OUT_LEN))
end

-- 5. Запуск обучения
print("Начинаю обучение (это может занять время)...")

local train_config = {
    epochs = 200,     -- Больше эпох для текста (нужно чтобы ID слов совпали)
    lr = 0.001,        -- Скорость обучения
    batch_size = 2,    -- Размер пачки данных
    output_size = OUT_LEN
}

-- Функция обратного вызова для отслеживания прогресса
local function onEpoch(epoch)
    if epoch % 100 == 0 then
        print("Epoch: " .. epoch)
    end
end

local success = djl.train(model_id, train_config, train_data, nil, onEpoch)

if success then
	print("Training completed successfully!")

	-- 6. Проверка работы перед сохранением
	local test_q = "What are you doing, how are you doing?"
	local encoded_q = ai:encode(test_q, IN_LEN)
	local prediction = djl.predict(model_id, encoded_q)
	
	print("---------------------------------")
	print("Test after training:")
	print("Question: " .. test_q)
	print("AI response: " .. ai:decode(prediction))
	print("---------------------------------")

	-- 7. СОХРАНЕНИЕ МОДЕЛИ И КОНФИГА
	-- Создаем папку 'models', если её нет (зависит от ОС)
	-- os.execute("mkdir models") 
	
	local save_path = "models/my_bot"
	if ai:save(save_path) then
		print("All model files are saved to the path: " .. save_path)
	else
		print("Error saving files!")
	end

else
	print("Error during training.")
end

-- 8. Закрываем модель
djl.close(model_id)
