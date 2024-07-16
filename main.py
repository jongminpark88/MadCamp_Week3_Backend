import os
import re
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from datetime import date, datetime
import motor.motor_asyncio
from bson import ObjectId
from openai import OpenAI
from dotenv import load_dotenv
import torch


load_dotenv()
app = FastAPI()


#openai.api_key = os.getenv('OPENAI_API_KEY')
client_gpt=OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# MongoDB 연결 설정
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://<username>:<password>@cluster0.mongodb.net/mydatabase?retryWrites=true&w=majority")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client.mydatabase



# Pydantic 모델 정의
class User(BaseModel):
    user_id: str
    password: str
    profile_image: str
    nickname: str
    birth: date
    book_list: List[str] = []

    @validator('birth', pre=True, always=True)
    def parse_birth(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        return value

    class Config:
        json_encoders = {
            date: lambda v: v.strftime('%Y-%m-%d'),
        }


# 새로운 UserUpdate 모델 정의
class UserUpdate(BaseModel):
    password: str
    profile_image: str
    nickname: str
    birth: date

    @validator('birth', pre=True, always=True)
    def parse_birth(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        return value

    class Config:
        json_encoders = {
            date: lambda v: v.strftime('%Y-%m-%d'),
        }




class Book(BaseModel):
    book_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    book_title: str
    book_cover_image: str
    page_list: List[str] = []
    book_creation_day: date
    owner_user: str
    book_private: bool
    book_theme: str

    @validator('book_creation_day', pre=True, always=True)
    def parse_creation_day(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        return value

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            date: lambda v: v.strftime('%Y-%m-%d'),
        }


# 새로운 BookUpdate 모델 정의
class BookUpdate(BaseModel):
    book_title: str
    book_cover_image: str
    book_creation_day: date
    book_private: bool
    book_theme: str

    @validator('book_creation_day', pre=True, always=True)
    def parse_creation_day(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        return value

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            date: lambda v: v.strftime('%Y-%m-%d'),
        }






class Page(BaseModel):
    page_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    page_title: str
    page_content: str
    page_creation_day: date
    owner_book: str
    owner_user: str
    book_theme: str

    @validator('page_creation_day', pre=True, always=True)
    def parse_creation_day(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        return value

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            date: lambda v: v.strftime('%Y-%m-%d'),
        }




class PageUpdate(BaseModel):
    page_title: str
    page_content: str
    page_creation_day: date

    @validator('page_creation_day', pre=True, always=True)
    def parse_creation_day(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        return value

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            date: lambda v: v.strftime('%Y-%m-%d'),
        }



class Sentence(BaseModel):
    sentence: str


class StoryRequest(BaseModel):
    user_story: str
    desired_style: str

class StoryResponse(BaseModel):
    title: str
    content: str


# 감성 분석 요청 모델 정의
class SentimentRequest(BaseModel):
    text: str

# 감성 분석 응답 모델 정의
class SentimentResponse(BaseModel):
    sentiment_score: int


# 사용자 정보 조회
@app.get("/user/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"user_id": user_id})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user['birth'] = user['birth'].date()  # datetime을 date로 변환
    return user

@app.get("/user/", response_model=List[User])
async def get_all_users():
    users = await db.users.find().to_list(1000)
    for user in users:
        user['birth'] = user['birth'].date()  # datetime을 date로 변환
    return users




# 책 정보 조회
@app.get("/book/user/{user_id}", response_model=List[Book])
async def get_books_by_user(user_id: str):
    books = await db.books.find({"owner_user": user_id}).to_list(1000)
    for book in books:
        book['book_creation_day'] = book['book_creation_day'].date()  # datetime을 date로 변환
    return books

@app.get("/book/{book_id}", response_model=Book)
async def get_book(book_id: str):
    book = await db.books.find_one({"_id": book_id})
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    book['book_creation_day'] = book['book_creation_day'].date()  # datetime을 date로 변환
    book['book_id'] = str(book['_id'])
    return book

@app.get("/book/theme/{book_theme}", response_model=List[Book])
async def get_books_by_theme(book_theme: str):
    books = await db.books.find({"book_theme": book_theme, "book_private": False}).to_list(1000)
    for book in books:
        book['book_creation_day'] = book['book_creation_day'].date()  # datetime을 date로 변환
    return books


# 페이지 정보 조회
@app.get("/page/book/{book_id}", response_model=List[Page])
async def get_pages_by_book(book_id: str):
    pages = await db.pages.find({"owner_book": book_id}).to_list(1000)
    for page in pages:
        page['page_creation_day'] = page['page_creation_day'].date()  # datetime을 date로 변환
    return pages


@app.get("/page/{page_id}", response_model=Page)
async def get_page(page_id: str):
    page = await db.pages.find_one({"_id": page_id})
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found")
    page['page_creation_day'] = page['page_creation_day'].date()  # datetime을 date로 변환
    page['page_id'] = str(page['_id'])
    return page

# 책 추가
@app.post("/book/{user_id}/insert", response_model=Book)
async def insert_book(user_id: str, book: Book):
    user = await db.users.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    book.owner_user = user_id
    book_dict = book.dict(by_alias=True)
    book_dict['book_creation_day'] = datetime.combine(book_dict['book_creation_day'], datetime.min.time())  # datetime으로 변환
    result = await db.books.insert_one(book_dict)
    await db.users.update_one({"user_id": user_id}, {"$push": {"book_list": book.book_id}})
    
    return book

# 페이지 추가
@app.post("/page/{book_id}/insert", response_model=Page)
async def insert_page(book_id: str, page: Page):
    book = await db.books.find_one({"_id": book_id})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    page.owner_book = book_id
    page_dict = page.dict(by_alias=True)
    page_dict['page_creation_day'] = datetime.combine(page_dict['page_creation_day'], datetime.min.time())  # datetime으로 변환
    result = await db.pages.insert_one(page_dict)
    await db.books.update_one({"_id": book_id}, {"$push": {"page_list": page.page_id}})
    
    return page

# 사용자 추가
@app.post("/user/insert", response_model=User)
async def insert_user(user: User):
    existing_user = await db.users.find_one({"user_id": user.user_id})
    if existing_user:
        raise HTTPException(status_code=400, detail="User ID already exists")
    
    user_dict = user.dict(by_alias=True)
    user_dict['birth'] = datetime.combine(user_dict['birth'], datetime.min.time())  # datetime으로 변환
    await db.users.insert_one(user_dict)
    return user

#책 삭제
@app.delete("/book/delete/{book_id}")
async def delete_book(book_id: str):
    book = await db.books.find_one({"_id": book_id})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    # 책에 속한 모든 페이지 삭제
    await db.pages.delete_many({"owner_book": book_id})
    
    # 사용자 데이터베이스의 book_list에서 책 ID 제거
    await db.users.update_one({"user_id": book['owner_user']}, {"$pull": {"book_list": book_id}})
    
    # 책 삭제
    await db.books.delete_one({"_id": book_id})
    
    return {"message": "Book and its pages deleted successfully"}


#페이지 삭제
@app.delete("/page/delete/{page_id}")
async def delete_page(page_id: str):
    page = await db.pages.find_one({"_id": page_id})
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    
    # 책 데이터베이스의 page_list에서 페이지 ID 제거
    await db.books.update_one({"_id": page['owner_book']}, {"$pull": {"page_list": page_id}})
    
    # 페이지 삭제
    await db.pages.delete_one({"_id": page_id})
    
    return {"message": "Page deleted successfully"}



#사용자 정보 수정
@app.put("/user/edit/{user_id}", response_model=User)
async def update_user(user_id: str, user_update: UserUpdate):
    existing_user = await db.users.find_one({"user_id": user_id})
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    # 현재의 book_list를 가져와 유지
    current_book_list = existing_user.get('book_list', [])

    update_data = user_update.dict(exclude_unset=True)
    update_data['book_list'] = current_book_list  # 기존 book_list를 유지
    
    if 'birth' in update_data:
        update_data['birth'] = datetime.combine(update_data['birth'], datetime.min.time())  # datetime으로 변환

    await db.users.update_one({"user_id": user_id}, {"$set": update_data})
    updated_user = await db.users.find_one({"user_id": user_id})
    updated_user['birth'] = updated_user['birth'].date()
    return updated_user


# 책 정보 수정
@app.put("/book/edit/{book_id}", response_model=Book)
async def update_book(book_id: str, book_update: BookUpdate):
    existing_book = await db.books.find_one({"_id": book_id})
    if not existing_book:
        raise HTTPException(status_code=404, detail="Book not found")

    # 현재의 book_id, page_list, owner_user 값을 가져와 유지
    current_book_id = existing_book.get('_id')
    current_page_list = existing_book.get('page_list', [])
    current_owner_user = existing_book.get('owner_user')

    update_data = book_update.dict(exclude_unset=True, by_alias=True)
    update_data['_id'] = current_book_id  # 기존 book_id 유지
    update_data['page_list'] = current_page_list  # 기존 page_list 유지
    update_data['owner_user'] = current_owner_user  # 기존 owner_user 유지
    
    if 'book_creation_day' in update_data:
        update_data['book_creation_day'] = datetime.combine(update_data['book_creation_day'], datetime.min.time())  # datetime으로 변환

    await db.books.update_one({"_id": book_id}, {"$set": update_data})
    updated_book = await db.books.find_one({"_id": book_id})
    updated_book['book_creation_day'] = updated_book['book_creation_day'].date()
    return updated_book



#페이지 정보 수정
@app.put("/page/edit/{page_id}", response_model=Page)
async def update_page(page_id: str, page_update: PageUpdate):
    existing_page = await db.pages.find_one({"_id": page_id})
    if not existing_page:
        raise HTTPException(status_code=404, detail="Page not found")

    # 현재의 book_id, page_list, owner_user 값을 가져와 유지
    current_page_id = existing_page.get('_id')
    current_owner_book = existing_page.get('owner_book')
    current_owner_user = existing_page.get('owner_user')
    current_book_theme = existing_page.get('book_theme')


    update_data = page_update.dict(exclude_unset=True, by_alias=True)
    update_data['_id'] = current_page_id  # 기존 book_id 유지
    update_data['owner_book'] = current_owner_book  # 기존 owner_book 유지
    update_data['owner_user'] = current_owner_user  # 기존 owner_user 유지
    update_data['book_theme'] = current_book_theme  # 기존 book_theme 유지

    if 'page_creation_day' in update_data:
        update_data['page_creation_day'] = datetime.combine(update_data['page_creation_day'], datetime.min.time())  # datetime으로 변환

    await db.pages.update_one({"_id": page_id}, {"$set": update_data})
    updated_page = await db.pages.find_one({"_id": page_id})
    updated_page['page_creation_day'] = updated_page['page_creation_day'].date()
    return updated_page



##############################################################################
##############################################################################
##############################################################################

#####gpt로 스토리 만들기#####
@app.post("/generate", response_model=StoryResponse)
async def transform_text(story_request: StoryRequest):
    user_story = story_request.user_story
    desired_style = story_request.desired_style

    if not user_story or not desired_style:
        raise HTTPException(status_code=400, detail="user_story and desired_style are required")

    try:
        # GPT API 호출하여 텍스트 변환
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"당신은 <{desired_style}>의 작가입니다."},
                {"role": "user", "content": f"다음 이야기를 <{desired_style}> 스타일로 자서전을 작성해주세요(소제목 없이 본문만): {user_story}"}
            ],
            max_tokens=1200,
            temperature=0.7,
        )

        transformed_story = response.choices[0].message.content.strip().replace('\\', '').replace('\n', '')


        # 제목을 생성하기 위해 추가 요청
        title_response = client_gpt.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"당신은 <{desired_style}>의 작가입니다."},
                {"role": "user", "content": f"다음 이야기를 <{desired_style}> 스타일로 자서전을 썼을때, 어울리는 제목 하나만 작성해주세요: {user_story}"}
            ],
            max_tokens=100,
            temperature=0.7,
        )

        title = title_response.choices[0].message.content.strip().replace('\\', '').replace('\n', '')


        return StoryResponse(title=title, content=transformed_story)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



##############################################################################
##############################################################################
##############################################################################


# GPT를 사용한 감성 분석 함수 정의
def analyze_sentiment_gpt(text: str) -> int:
    response = client_gpt.chat.completions.create(
        model="gpt-3.5-turbo",  # GPT-3.5 모델 사용
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"다음 문장의 감성을 분석하고, 긍정적일수록 100점, 부정적일수록 0점으로 평가하세요(점수만 출력해):\n\n\"{text}\"\n\n감성 점수:"}
        ],
        max_tokens=5,
        temperature=0.0,
    )
    sentiment_score = int(response.choices[0].message.content.strip())
    return sentiment_score

# 감성 분석 API 엔드포인트 정의
@app.post("/sentiment/", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        sentiment_score = analyze_sentiment_gpt(request.text)
        return SentimentResponse(sentiment_score=sentiment_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
