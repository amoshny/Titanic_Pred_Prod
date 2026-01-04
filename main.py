from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import pickle
from pydantic import BaseModel

# Инициализация FastAPI приложения
app = FastAPI()

# Загрузка модели при старте приложения
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Счетчик запросов
request_counter = 0

# Определение модели данных для входящих запросов, используя Pydantic(BaseModel) -> то есть описание структуры данных
class TitanicData(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float

# Эндпоинт для получения счетчика запросов
@app.get("/stats")
def stats():
    """Эндпоинт для получения статистики по количеству запросов."""
    return {"request_count": request_counter}

# Эндпоинт для проверки состояния сервиса
@app.get("/health")
def health():
    """Эндпоинт для проверки состояния сервиса."""
    return {"status": "ok"}

# Эндпоинт для предсказания выживаемости пассажира
@app.post("/predict_titanic") # @ - декоратор, указывающий, что функция ниже будет обрабатывать POST-запросы по указанному пути
def predict_titanic(data: TitanicData):
    """Эндпоинт для предсказания выживаемости пассажира."""
    global request_counter # Объявление глобальной переменной для счетчика запросов, global позволяет изменять глобальную переменную внутри функции
    request_counter += 1 # Увеличение счетчика запросов при каждом вызове эндпоинта

    # создание DataFrame из входных данных
    new_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],
    'Age': [22.0],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [2]})

    # выполнение предсказания
    prediction = model.predict(new_data)
    return 'Prediction: Survived' if prediction[0] == 1 else 'Prediction: Did not survive'



# Запуск приложения (если нужно запустить локально)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)