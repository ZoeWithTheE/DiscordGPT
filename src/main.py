import discord
from discord import option
from discord.ext import commands

import os
import re
import json
import asyncio as aio
from datetime import datetime, timedelta

import openai as ai
from transformers import GPT2TokenizerFast

import yaml
from dotenv import load_dotenv

def config(key: str): return yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "environment", "config.yml"))).get(key)
file_paths = config('file_paths')
replacements = config('completion_replacements')
gpt_model = config('gpt_model')
max_tokens = config('max_tokens')
forced_chat_method = config('forced_chat_method')
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), file_paths['dotenv']))
def secret(secret: str): return os.getenv(secret)

client = commands.Bot(intents=discord.Intents.all())

async def continuous_typing(channel, stop_event):
    while not stop_event.is_set():
        async with channel.typing(): await aio.sleep(8)

async def generate(channel, convo):
    stop_event = aio.Event()
    typing_task = aio.create_task(continuous_typing(channel, stop_event))
    try:
        loop = aio.get_event_loop()
        completion = await loop.run_in_executor(None, lambda: ai.ChatCompletion.create(model=gpt_model, messages=convo, max_tokens=max_tokens))
        completion = completion.choices[0].message["content"]
    finally:
        stop_event.set()
        await typing_task
        typing_task.cancel()
    for replacement in replacements:
        completion = completion.replace(replacement[0], replacement[1])
    return completion

ai.api_key = secret("OPENAI_API_KEY")
def getTokens(t:str):return sum([len(GPT2TokenizerFast.from_pretrained("gpt2")(m)['input_ids']) for m in (t[i:i+1024] for i in range(0,len(t),1024))])

class UserData:
    def __init__(self, member: discord.Member, path=os.path.join(os.path.dirname(__file__), file_paths['user_data'])):
        self.path = path
        self.member = member
        self.user_id = member.id

    async def load_data(self):
        with open(self.path, 'r') as f:
            return json.load(f)

    async def save_data(self, data):
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=4)

    async def create_user(self):
        data = await self.load_data()
        default_user = next((user for user in data["users"] if user["id"] == 0), None)
        if default_user is None: 
            raise ValueError("Default user not found")
        if any(user["id"] == self.user_id for user in data["users"]): 
            return None
        new_user = default_user.copy()
        new_user["id"] = self.user_id
        data["users"].append(new_user)
        await self.save_data(data)

    async def set_user_value(self, key, value=None):
        data = await self.load_data()
        user = next((user for user in data["users"] if user["id"] == self.user_id), None)
        default_user = next((user for user in data["users"] if user["id"] == 0), None)
        if default_user is None: 
            raise ValueError("Default user not found")
        if user is None:
            await self.create_user()
            user = default_user.copy()
            user["id"] = self.user_id
        if key not in user and key in default_user: 
            user[key] = default_user[key]
        if value is not None: 
            user[key] = value
        await self.save_data(data)

    async def delete_user_value(self, key):
        data = await self.load_data()
        user = next((user for user in data["users"] if user["id"] == self.user_id), None)
        if user is None:
            raise ValueError("User not found")
        if key not in user:
            raise ValueError("Key not found in user data")
        del user[key]
        await self.save_data(data)

    async def get_user_value(self, key):
        data = await self.load_data()
        user = next((user for user in data["users"] if user["id"] == self.user_id), None)
        default_user = next((user for user in data["users"] if user["id"] == 0), None)
        if default_user is None: 
            raise ValueError("Default user not found")
        if user is None:
            await self.create_user()
            user = default_user.copy()
            user["id"] = self.user_id
        if key not in user and key in default_user: 
            user[key] = default_user[key]
        return user[key]

    async def delete_user(self):
        data = await self.load_data()
        user = next((user for user in data["users"] if user["id"] == self.user_id), None)
        if user is None: 
            raise ValueError("User not found")
        data["users"].remove(user)
        await self.save_data(data)

@client.slash_command(name='chat-method', description='Interact via threads or replying')
@option('chat_method', description='Set the chat method to THREAD or REPLY', choices=['THREAD', 'REPLY'], required=True)
async def set(ctx: discord.ApplicationContext, chat_method):
    if forced_chat_method['enabled']:
        chat_method = forced_chat_method['value']
        await ctx.response.send_message(f'Sorry, but this feature has been locked! The chat method will remain \"{chat_method}\".', ephemeral=True)
    await UserData(ctx.author).set_user_value('interaction_method', chat_method)
    await ctx.response.send_message(f'Set chat method to \"{chat_method}\".', ephemeral=True)

@client.slash_command(name='set-personality', description='Set your interaction personality')
@option('personality', description='Choose a personality', choices=[name for name in config('personalities').keys() if name != 'default'], required=True)
async def set_personality(ctx: discord.ApplicationContext, personality):
    await UserData(ctx.author).set_user_value('personality', personality)
    await ctx.response.send_message(f'Set personality to \"{personality}\".', ephemeral=True)

def remove_formatting(message):
    formatting_symbols = ['*', '_', '~', '#', '`']
    for symbol in formatting_symbols: message = message.replace(symbol, '')
    return message

async def get_user_personality(user: discord.User):
    user_personality = await UserData(user).get_user_value('personality')
    if not user_personality: user_personality = config('personalities')[config('personalities')['DEFAULT']]
    if user_personality == "DEFAULT": user_personality = config('personalities')[config('personalities')['DEFAULT']]
    else: user_personality = config('personalities')[user_personality]
    return user_personality.format(
        # Add your own variables here
    )

def extract_command_and_parameters(message: str):
    match = re.search(r'START:(.*):END', message)
    if match:  return match.group(1)
    else: return None

async def execute_encoded_message(message, member: discord.Member, convo):
    full_executor = extract_command_and_parameters(message)
    if not full_executor: return message, convo  # Return the convo unaltered if there's no command to execute
    split_executor = full_executor.split(">")
    command = split_executor[0].lower()
    parameters = split_executor[1:]

    
    # Commands that GPT can send to either get information about a user/the server, or perform actions
    # Example system prompt to teach GPT how to use:

    ## You can use commands. These commands will be removed from your message before it is sent.
    ## Commands you can use are: BAN (ban a user), MUTE (mute a user), MEMBERCOUNT (get the server membercount), ROLES (get the roles of a user).
    ## For BAN: parameter 1 = Discord member ID, parameter 2 = ban reason.
    ## For MUTE: parameter 1 = Discord member ID, parameter 2 = mute reason, parameter 3 = mute duration (seconds).
    ## For MEMBERCOUNT: no parameters.
    ## For ROLES: parameter 1 = Discord member ID.
    ## Command format: START:{COMMAND}>{PARAMETER n}
    ## Example ban command: START:ban>968356025461768192>put your reason here:END
    ## Example mute command (1 minute): START:mute>968356025461768192>put your reason here>60:END
    ## Example membercount command: START:membercount:END

    # Use with great care, and always implement a permissions check if you don't want any user to be able to request any action

    """
    if command.lower() == 'ban':
        if member.guild_permissions.ban_members:
            try:
                user_id_to_ban = int(parameters[0])
                ban_reason = parameters[1]
                user_to_ban = await client.fetch_user(user_id_to_ban)
                if user_to_ban:
                    await member.guild.ban(user_to_ban, reason=ban_reason)
                    convo.append({"content": "operation complete", "role": "system"})
            except Exception as e: convo.append({"content": f"operation got an error: {e}. Relay this error to the user.", "role": "system"})
        else: convo.append({"content": "The member who asked you to ban that user lacks the permission to request such an action", "role": "system"})
    """
    
    return message.replace('START:' + full_executor + ':END', ''), convo

def save_to_transcript(first_message: discord.Message, convo):
    transcript_path = os.path.join(os.path.dirname(__file__), file_paths['transcript'])
    with open(transcript_path, 'r') as f: transcript = json.load(f)
    total_tokens = sum(getTokens(message['content']) for message in convo)
    transcript_id = first_message.id
    author_id = first_message.author.id
    existing_transcript = next((entry for entry in transcript if entry["transcript_id"] == transcript_id), None)
    if existing_transcript:
        existing_transcript["tokens"] += total_tokens
        existing_transcript["conversation"]["message_count"] += len(convo)
        existing_transcript["conversation"]["messages"] = convo
    else:
        new_entry = {
            "transcript_id": transcript_id,
            "author_id": author_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "tokens": total_tokens,
            "conversation": {
                "message_count": len(convo),
                "messages": convo
            }
        }
        transcript.append(new_entry)
    with open(transcript_path, 'w') as f: json.dump(transcript, f, indent=4)

@client.event
async def on_ready(): print(f'\n\nSuccessfully logged into Discord as {client.user}')

@client.event
async def on_message(ctx):
    author = ctx.author
    authorID = author.id
    triggers = config('triggering')['triggers']
    ignore_modifiers = config('triggering')['ignore_modifiers']
    ignore_case = config('triggering')['ignore_case']
    message_content = ctx.content
    if ignore_modifiers: trigger_message_content = remove_formatting(message_content)
    if ignore_case:
        trigger_message_content = message_content.lower()
        triggers = [trigger.lower() for trigger in triggers]
    if any(trigger_message_content.startswith(trigger) for trigger in triggers) and not author.bot:
        if (await UserData(author).get_user_value('interaction_method') == "THREAD") and not isinstance(ctx.channel, discord.channel.DMChannel):
            now = datetime.now()
            title = config('message_formats')['thread_title'].format(
                user = author.name,
                time = now.strftime("%H:%M:%S"),
                date = now.strftime("%Y-%m-%d")
                )
            await ctx.create_thread(name=title, auto_archive_duration=60)
            thread = ctx.thread
            convo = []
            convo.append({"content": await get_user_personality(author), "role": "system"})
            convo.append({"content": message_content, "role": "user"})
            response = await generate(thread, convo)
            convo.append({"content": response, "role": "assistant"})
            save_to_transcript(ctx, convo)
            response, convo = await execute_encoded_message(response, author, convo)
            for m in (response[i:i+2000] for i in range(0, len(response), 2000)): await thread.send(m)
            if convo[-1]["role"] == "system":
                response = await generate(thread, convo)
                convo.append({"content": response, "role": "assistant"})
                save_to_transcript(ctx, convo)
                for m in (response[i:i+2000] for i in range(0, len(response), 2000)): await thread.send(m)
            conversationHappening = True
            def check(m):return m.content is not None and m.channel.id == thread.id
            while conversationHappening:
                try: userMessage = await client.wait_for("message", timeout=1200, check=check)
                except:
                    conversationHappening = False
                    await thread.delete()
                if userMessage:
                    if userMessage.author.id == authorID:
                        convo.append({"content": userMessage.content, "role": "user"})
                        response = await generate(thread, convo)
                        convo.append({"content": response, "role": "assistant"})
                        save_to_transcript(ctx, convo)
                        response, convo = await execute_encoded_message(response, author, convo)
                        for m in (response[i:i+2000] for i in range(0, len(response), 2000)): await thread.send(m)
                        if convo[-1]["role"] == "system":
                            response = await generate(thread, convo)
                            convo.append({"content": response, "role": "assistant"})
                            save_to_transcript(ctx, convo)
                            for m in (response[i:i+2000] for i in range(0, len(response), 2000)): await thread.send(m)
        elif (await UserData(author).get_user_value('interaction_method') == "REPLY") or isinstance(ctx.channel, discord.channel.DMChannel):
            convo = []
            convo.append({"content": await get_user_personality(author), "role": "system"})
            convo.append({"content": message_content, "role": "user"})
            response = await generate(ctx.channel, convo)
            convo.append({"content": response, "role": "assistant"})
            save_to_transcript(ctx, convo)
            response, convo = await execute_encoded_message(response, author, convo)
            for m in (response[i:i+2000] for i in range(0, len(response), 2000)): lastMessage = await ctx.reply(m,mention_author=False)
            if convo[-1]["role"] == "system":
                response = await generate(ctx.channel, convo)
                convo.append({"content": response, "role": "assistant"})
                save_to_transcript(ctx, convo)
                for m in (response[i:i+2000] for i in range(0, len(response), 2000)): lastMessage = await ctx.reply(m,mention_author=False)
            conversationHappening = True
            def check(m):
                if (m.content is not None and m.reference is not None) and (m.reference.message_id == lastMessage.id): return True
                return False
            while conversationHappening:
                try: userMessage = await client.wait_for("message", timeout=1200, check=check)
                except: conversationHappening = False
                if userMessage:
                    if userMessage.author.id == authorID:
                        convo.append({"content": userMessage.content, "role": "user"})
                        response = await generate(ctx.channel, convo)
                        convo.append({"content": response, "role": "assistant"})
                        save_to_transcript(ctx, convo)
                        response, convo = await execute_encoded_message(response, author, convo)
                        for m in (response[i:i+2000] for i in range(0, len(response), 2000)): lastMessage = await ctx.reply(m,mention_author=False)
                        if convo[-1]["role"] == "system":
                            response = await generate(ctx.channel, convo)
                            convo.append({"content": response, "role": "assistant"})
                            save_to_transcript(ctx, convo)
                            for m in (response[i:i+2000] for i in range(0, len(response), 2000)): lastMessage = await ctx.reply(m,mention_author=False)

client.run(secret("DISCORD_BOT_TOKEN"))
