import atexit

from langchain_core.tools import tool
from pathlib import Path

from datebase.mock_db import get_connection, query_db, close_db

db_conn = get_connection()
atexit.register(close_db, db_conn)      # 注册进程退出时的清理钩子hook ，只要不是断电、kill -9 都可以自动关闭数据库资源

# 预留文件输出
# OUTPUT_DIR = Path(__file__).parent.parent / "output_certs"
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

@tool
def get_employee_profile(uid:str) -> str:
    """
    根据员工 UID 查询员工的完整人事档案，包括姓名、职级、工作城市、入职年限、基本性质。
    当需要获取当前对话员工的背景属性时，必须首先调用此工具
    """
    sql = "select uid, name, rank, location, seniority, base_salary from employees where uid = ?"
    res = query_db(conn=db_conn, sql=sql, params=(uid,))

    if not res:
        return f"「错误」：未找到 UID 为 {uid} 的员工信息"

    employee = res[0]
    return (f'「档案查询结果」员工姓名：{employee['name']}，职级：{employee['rank']}，'
            f'工作地点：{employee['location']}，入职年限：{employee['seniority']} 年，'
            f'基本薪资：{employee['base_salary']} 元。')

@tool
def get_leave_balance(uid:str) -> str:
    """
    根据员工 UID 查询剩余的假期余额（包括年假和病假）。
    当员工明确询问 ‘我还有几天假’ 或 ‘我的假期余额’ 时调用。
    """
    sql = """
    SELECT a.name,
           b.annual_leave_remaining,
           b.sick_leave_remaining
    from employees a
    LEFT JOIN leave_balances b on a.uid = b.uid
    where a.uid = ?
    """
    res = query_db(conn=db_conn, sql=sql, params=(uid,))

    if not res:
        return f"「错误」：无法获取 UID 为 {uid} 的假期数据。"

    data = res[0]
    return (f"「假期系统」员工 {data['name']}(UID:{uid}) 当前剩余法定/福利年假：{data['annual_leave_remaining']} 天，"
            f"剩余带薪病假：{data['sick_leave_remaining']} 天。")

@tool
def generate_employment_certificate(uid:str, cer_type:str) -> str:
    """
    为指定员工自动生成证明文件。
    参数 cer_type必须是以下两个值之一：
    - 'employment':仅开具在职证明（全员可用）
    - 'income': 开具包含薪资的在职收入正面（有职阶权限限制，仅p5及以上可以使用）
    :param uid:
    :param cer_type:
    :return:
    """
    sql = "select name, rank, base_salary from employees where uid = ?"
    emp_res = query_db(conn=db_conn, sql=sql, params=(uid,))

    if not emp_res:
        return f'因无法核实员工身份(UID: {uid}), 证明生成失败'
    employee = emp_res[0]

    if cer_type == 'income':
        try:
            rank_level = int(employee['rank'].lower().replace('p', ''))
        except ValueError:
            rank_level = 0

        if rank_level < 5:
            return (f"「系统提示」根据公司规定，P4 及以下职级员工（当前员工为 {employee['rank']}）无法线上开具薪资收入证明。"
                    f"请引导员工在线提交人工工单，由 HR 线下手动核实开具。")

        content = (f"《薪资收入证明》\n 兹证明我公司员工 {employee['name']} ，职级为 {employee['rank']}。\n"
                   f"该员工基本薪资为人民币 {employee['base_salary']} 元。\n 特此证明 （公章）")

        # 如果需要保存文件，可以使用 pathlib
        # file_path = OUTPUT_DIR / f'{employee["name"]}_income_cert.txt'
        # file_path.write_text(content, encoding="utf-8")

        return (f"「系统成功」已自动为您生成收入证明：\n --------------------------\n"
                f"{content}"
                f"\n--------------------------")

    elif cer_type == 'employment':

        content = (f"《在职证明》\n兹证明 {employee['name']} 现为我公司在职员工，职级为 {employee['rank']} 。特此证明。"
                   f"\n      （公章）")
        return (f"「系统成功」已自动为您生成在职证明：\n --------------------------\n"
                f"{content}"
                f"\n--------------------------")

    return "错误：不支持的证明类型。可选类型为 'employment' 或 'income'"






