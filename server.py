from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime, timezone
import uuid
import os
import logging
import base64
import bcrypt
import stripe
from contextlib import asynccontextmanager

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ['MONGO_URL']
DB_NAME = os.environ['DB_NAME']
STRIPE_API_KEY = os.environ.get('STRIPE_API_KEY')
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Stripe
stripe.api_key = STRIPE_API_KEY

# Lifespan for FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code here
    logger.info("Starting up FastAPI app")
    yield
    # Shutdown code here
    logger.info("Shutting down FastAPI app")
    client.close()

app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# Models


class Product(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    price: float
    category: str
    images: List[str] = []
    stock: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


class ProductCreate(BaseModel):
    name: str
    description: str
    price: float
    category: str
    images: List[str] = []
    stock: int = 0


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    images: Optional[List[str]] = None
    stock: Optional[int] = None


class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: str
    is_admin: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


class UserRegister(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class CartItem(BaseModel):
    product_id: str
    quantity: int = 1


class Cart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: str
    items: List[CartItem] = []
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


class Order(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    email: str
    items: List[Dict]
    total: float
    status: str = "pending"
    payment_session_id: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


class PaymentTransaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    amount: float
    currency: str
    status: str = "pending"
    payment_status: str = "unpaid"
    metadata: Optional[Dict] = {}
    order_id: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


class CheckoutRequest(BaseModel):
    email: str
    items: List[Dict]
    origin_url: str

# Helper functions


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Routes


@app.get("/")
async def root_app():
    return {"message": "Luxury E-commerce API is running"}


@api_router.get("/")
async def root():
    return {"message": "Luxury E-commerce API"}

# Products


@api_router.get("/products", response_model=List[Product])
async def get_products(category: Optional[str] = None):
    query = {}
    if category:
        query["category"] = category
    products = await db.products.find(query, {"_id": 0}).to_list(1000)
    for p in products:
        if isinstance(p.get("created_at"), str):
            p["created_at"] = datetime.fromisoformat(p["created_at"])
    return products


@api_router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str):
    product = await db.products.find_one({"id": product_id}, {"_id": 0})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    if isinstance(product.get("created_at"), str):
        product["created_at"] = datetime.fromisoformat(product["created_at"])
    return product


@api_router.post("/products", response_model=Product)
async def create_product(product: ProductCreate):
    prod = Product(**product.model_dump())
    doc = prod.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.products.insert_one(doc)
    return prod


@api_router.put("/products/{product_id}", response_model=Product)
async def update_product(product_id: str, product_update: ProductUpdate):
    update_data = {k: v for k,
                   v in product_update.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await db.products.update_one({"id": product_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    updated_product = await db.products.find_one({"id": product_id}, {"_id": 0})
    if isinstance(updated_product.get("created_at"), str):
        updated_product["created_at"] = datetime.fromisoformat(
            updated_product["created_at"])
    return updated_product


@api_router.delete("/products/{product_id}")
async def delete_product(product_id: str):
    result = await db.products.delete_one({"id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"message": "Product deleted successfully"}

# Image upload


@api_router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"
        return {"url": f"data:{mime_type};base64,{base64_image}"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading image: {str(e)}")

# Categories


@api_router.get("/categories")
async def get_categories():
    return [
        {"id": "insectes-eclates", "name": "Insectes éclatés"},
        {"id": "insectes", "name": "Insectes"},
        {"id": "squelettes-eclates", "name": "Squelettes éclatés"}
    ]

# Auth


@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_data.email,
                password_hash=hash_password(user_data.password))
    doc = user.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.users.insert_one(doc)
    return {"id": user.id, "email": user.email, "is_admin": user.is_admin}


@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"id": user["id"], "email": user["email"], "is_admin": user["is_admin"]}

# Cart


@api_router.post("/cart/add")
async def add_to_cart(item: CartItem, session_id: str = Header(default=None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    cart = await db.carts.find_one({"session_id": session_id}, {"_id": 0})
    if cart:
        items = cart.get("items", [])
        found = False
        for ci in items:
            if ci["product_id"] == item.product_id:
                ci["quantity"] += item.quantity
                found = True
                break
        if not found:
            items.append(item.model_dump())
        await db.carts.update_one({"session_id": session_id}, {"$set": {"items": items, "updated_at": datetime.now(timezone.utc).isoformat()}})
    else:
        new_cart = Cart(session_id=session_id, items=[item])
        doc = new_cart.model_dump()
        doc["updated_at"] = doc["updated_at"].isoformat()
        await db.carts.insert_one(doc)
    return {"message": "Item added to cart", "session_id": session_id}


@api_router.get("/cart")
async def get_cart(session_id: str = Header(default=None)):
    if not session_id:
        return {"items": []}
    cart = await db.carts.find_one({"session_id": session_id}, {"_id": 0})
    return cart or {"items": []}


@api_router.delete("/cart/remove/{product_id}")
async def remove_from_cart(product_id: str, session_id: str = Header(default=None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID")
    cart = await db.carts.find_one({"session_id": session_id}, {"_id": 0})
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found")
    items = [i for i in cart.get("items", []) if i["product_id"] != product_id]
    await db.carts.update_one({"session_id": session_id}, {"$set": {"items": items, "updated_at": datetime.now(timezone.utc).isoformat()}})
    return {"message": "Item removed from cart"}


@api_router.post("/cart/clear")
async def clear_cart(session_id: str = Header(default=None)):
    if not session_id:
        return {"message": "No cart to clear"}
    await db.carts.update_one({"session_id": session_id}, {"$set": {"items": [], "updated_at": datetime.now(timezone.utc).isoformat()}})
    return {"message": "Cart cleared"}

# Checkout


@api_router.post("/checkout/session")
async def create_checkout_session(checkout_req: CheckoutRequest):
    try:
        # Calculate total
        total = 0.0
        for item in checkout_req.items:
            product = await db.products.find_one({"id": item["product_id"]}, {"_id": 0})
            if not product:
                raise HTTPException(
                    status_code=404, detail=f"Product {item['product_id']} not found")
            total += product["price"] * item["quantity"]

        # Create order
        order = Order(email=checkout_req.email,
                      items=checkout_req.items, total=total)
        order_doc = order.model_dump()
        order_doc["created_at"] = order_doc["created_at"].isoformat()
        await db.orders.insert_one(order_doc)

        # Stripe session
        host_url = checkout_req.origin_url
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'eur',
                    'product_data': {'name': f"Order {order.id}"},
                    'unit_amount': int(total * 100),
                },
                'quantity': 1
            }],
            mode='payment',
            success_url=f"{host_url}/payment-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{host_url}/cart",
            metadata={"order_id": order.id, "email": checkout_req.email}
        )

        # Payment transaction
        payment_transaction = PaymentTransaction(
            session_id=session.id,
            amount=total,
            currency="eur",
            status="pending",
            payment_status="unpaid",
            metadata={"order_id": order.id, "email": checkout_req.email},
            order_id=order.id,
            email=checkout_req.email
        )
        payment_doc = payment_transaction.model_dump()
        payment_doc["created_at"] = payment_doc["created_at"].isoformat()
        payment_doc["updated_at"] = payment_doc["updated_at"].isoformat()
        await db.payment_transactions.insert_one(payment_doc)

        # Update order with session id
        await db.orders.update_one({"id": order.id}, {"$set": {"payment_session_id": session.id}})

        return {"url": session.url, "session_id": session.id}

    except Exception as e:
        logger.error(f"Checkout session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/checkout/status/{session_id}")
async def get_checkout_status(session_id: str):
    # Cherche la transaction locale
    payment = await db.payment_transactions.find_one(
        {"session_id": session_id}, {"_id": 0}
    )
    if not payment:
        raise HTTPException(
            status_code=404, detail="Payment session not found")

    # Si déjà payé localement, retourne directement
    if payment.get("payment_status") == "paid":
        return {
            "payment_status": "paid",
            "amount_total": payment.get("amount"),
            "currency": payment.get("currency"),
        }

    # Sinon, interroge Stripe pour avoir le statut actuel
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        stripe_status = session.payment_status  # "paid", "unpaid", "no_payment"
    except Exception as e:
        logger.error(f"Error retrieving Stripe session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Stripe session retrieval failed")

    # Si Stripe indique que le paiement est fait, mettre à jour Mongo
    if stripe_status == "paid":
        await db.payment_transactions.update_one(
            {"session_id": session_id},
            {"$set": {"payment_status": "paid", "status": "completed"}}
        )
        order_id = payment.get("order_id")
        if order_id:
            await db.orders.update_one(
                {"id": order_id},
                {"$set": {"status": "paid"}}
            )

    return {
        "payment_status": stripe_status,
        "amount_total": payment.get("amount"),
        "currency": payment.get("currency"),
    }


@api_router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail="Webhook error")

    # Écouter l'événement checkout.session.completed
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session.get("id")
        order_id = session["metadata"].get("order_id")

        # Mettre à jour payment_transaction
        await db.payment_transactions.update_one(
            {"session_id": session_id},
            {"$set": {"payment_status": "paid", "status": "completed"}}
        )
        # Mettre à jour l'order
        await db.orders.update_one(
            {"id": order_id},
            {"$set": {"status": "paid"}}
        )
        logger.info(f"Payment completed for session {session_id}")

    return {"status": "success"}

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)
