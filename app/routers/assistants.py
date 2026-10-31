from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Assistant
from app.schemas import AssistantCreate, AssistantOut, AssistantUpdate

router = APIRouter(prefix="/api/assistants", tags=["assistants"])


async def _get_or_404(session: AsyncSession, assistant_id: str) -> Assistant:
    assistant = await session.get(Assistant, assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return assistant


@router.get("", response_model=list[AssistantOut])
async def list_assistants(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Assistant).order_by(Assistant.created_at))
    return result.scalars().all()


@router.post("", response_model=AssistantOut, status_code=201)
async def create_assistant(data: AssistantCreate, session: AsyncSession = Depends(get_session)):
    assistant = Assistant(**data.model_dump())
    session.add(assistant)
    await session.commit()
    await session.refresh(assistant)
    return assistant


@router.get("/{assistant_id}", response_model=AssistantOut)
async def get_assistant(assistant_id: str, session: AsyncSession = Depends(get_session)):
    return await _get_or_404(session, assistant_id)


@router.patch("/{assistant_id}", response_model=AssistantOut)
async def update_assistant(assistant_id: str, data: AssistantUpdate, session: AsyncSession = Depends(get_session)):
    assistant = await _get_or_404(session, assistant_id)
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(assistant, field, value)
    await session.commit()
    await session.refresh(assistant)
    return assistant


@router.delete("/{assistant_id}", status_code=204)
async def delete_assistant(assistant_id: str, session: AsyncSession = Depends(get_session)):
    assistant = await _get_or_404(session, assistant_id)
    await session.delete(assistant)
    await session.commit()
    return Response(status_code=204)
